"""
Cross-attention policy network and value network.

PolicyNetwork
-------------
Architecture:
  1. Project h_obs, h_pool, h_pb each → dim=128, forming 3 state tokens (K, V).
  2. For each candidate x, project query(x) = [z_x; u(x); d_Dt; d_B] → dim=128.
  3. One cross-attention layer: candidate queries attend to 3 state tokens.
  4. Two-layer MLP head → scalar logit per candidate.
  5. Masked softmax over remaining (unselected) candidates.
  6. Training: sample from distribution. Evaluation: greedy argmax.

ValueNetwork
------------
  Shared linear encoder for [h_obs ; h_pool ; h_pb] → scalar V(s_{t,b}).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNetwork(nn.Module):
    """
    Cross-attention policy that selects one candidate per step.

    Parameters
    ----------
    h_obs_dim  : dimension of h_obs block.
    h_pool_dim : dimension of h_pool block.
    h_pb_dim   : dimension of h_pb block.
    query_dim  : dimension of candidate query (z_x + 3 scalars).
    attn_dim   : projected dimension for cross-attention (default 128).
    """

    def __init__(
        self,
        h_obs_dim: int,
        h_pool_dim: int,
        h_pb_dim: int,
        query_dim: int,
        attn_dim: int = 128,
    ):
        super().__init__()
        self.attn_dim = attn_dim

        # Project each state block to attn_dim (keys and values)
        self.proj_obs = nn.Linear(h_obs_dim, attn_dim)
        self.proj_pool = nn.Linear(h_pool_dim, attn_dim)
        self.proj_pb = nn.Linear(h_pb_dim, attn_dim)

        # Project candidate query to attn_dim
        self.proj_query = nn.Linear(query_dim, attn_dim)

        # Scaled dot-product attention scale factor
        self.scale = attn_dim ** -0.5

        # MLP head: (attn_dim + attn_dim) -> 128 -> 1
        # The cross-attention output is concatenated with the original query proj.
        self.head = nn.Sequential(
            nn.Linear(attn_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        h_obs: torch.Tensor,      # [B_env, h_obs_dim]  (B_env=1 usually)
        h_pool: torch.Tensor,     # [B_env, h_pool_dim]
        h_pb: torch.Tensor,       # [B_env, h_pb_dim]
        query_features: torch.Tensor,  # [B_env, P, query_dim]
        mask: Optional[torch.Tensor] = None,  # [B_env, P] bool, True=mask out
    ) -> torch.Tensor:
        """
        Returns logits [B_env, P].  Masked positions receive -inf.
        """
        B = h_obs.shape[0]
        P = query_features.shape[1]

        # State tokens: [B, 3, attn_dim]  (keys and values are the same here)
        tok_obs = self.proj_obs(h_obs).unsqueeze(1)    # [B, 1, attn_dim]
        tok_pool = self.proj_pool(h_pool).unsqueeze(1)  # [B, 1, attn_dim]
        tok_pb = self.proj_pb(h_pb).unsqueeze(1)       # [B, 1, attn_dim]
        state_tokens = torch.cat([tok_obs, tok_pool, tok_pb], dim=1)  # [B, 3, attn_dim]

        # Candidate queries: [B, P, attn_dim]
        q = self.proj_query(query_features)  # [B, P, attn_dim]

        # Cross-attention: queries (candidates) attend to state tokens
        # attn_weights [B, P, 3]
        attn_weights = (q @ state_tokens.transpose(-2, -1)) * self.scale  # [B, P, 3]
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Context vector per candidate: [B, P, attn_dim]
        context = attn_weights @ state_tokens  # [B, P, attn_dim]

        # Concatenate with query projection for skip connection
        combined = torch.cat([q, context], dim=-1)  # [B, P, 2*attn_dim]

        # Scalar logit per candidate
        logits = self.head(combined).squeeze(-1)  # [B, P]

        if mask is not None:
            logits = logits.masked_fill(mask, float("-inf"))

        return logits

    @torch.no_grad()
    def act(
        self,
        h_obs: torch.Tensor,
        h_pool: torch.Tensor,
        h_pb: torch.Tensor,
        query_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        greedy: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample (or greedy-select) one action.

        Returns
        -------
        action   : int tensor [B_env], index of selected candidate.
        log_prob : log probability of that action [B_env].
        """
        logits = self.forward(h_obs, h_pool, h_pb, query_features, mask)
        dist = torch.distributions.Categorical(logits=logits)
        if greedy:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob

    def log_prob_and_entropy(
        self,
        h_obs: torch.Tensor,
        h_pool: torch.Tensor,
        h_pb: torch.Tensor,
        query_features: torch.Tensor,
        actions: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute log π(a|s) and entropy for PPO update.

        Returns
        -------
        log_prob : [B_ppo]
        entropy  : [B_ppo]
        """
        logits = self.forward(h_obs, h_pool, h_pb, query_features, mask)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy()


class ValueNetwork(nn.Module):
    """
    Value baseline V(s_{t,b}).

    Takes the concatenated state [h_obs ; h_pool ; h_pb] and outputs a scalar.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        state : [B, state_dim]

        Returns
        -------
        torch.Tensor : [B]
        """
        return self.net(state).squeeze(-1)
