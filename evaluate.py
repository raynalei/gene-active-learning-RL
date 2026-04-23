"""
evaluate.py — Offline fair evaluation of a trained RL policy.

Loads a trained policy checkpoint, resets the predictor to **random
initialisation** (does NOT load predictor weights from the checkpoint),
then runs ONE clean AL episode.  Output CSV matches baseline.py so curves
can be plotted on the same axes.

Why a fresh predictor?
  During PPO training the ensemble accumulates fine-tuning across all 100
  episodes, so the predictor in the final checkpoint is NOT at round-0
  initialisation.  For a fair comparison against baseline.py (which also
  starts from a random predictor), we rebuild the ensemble from scratch and
  only restore the policy / value-net weights.

Usage
-----
python evaluate.py \\
    --checkpoint  results/policy_final.pt \\
    --gene_embeddings path/to/gene_embs.npy \\
    --cell_embeddings path/to/cell_embs.npy \\
    --h5ad            path/to/norman2019.h5ad \\
    [--config         path/to/config.yaml]    \\
    [--greedy]                                \\
    [--n_seeds 3]                             \\
    [--output_dir     eval_results/]
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import yaml

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from predict import sanitize_gene_embeddings, get_cached_num_guides, _encode_guide_merged
from predictor.ensemble import EnsemblePredictor
from predictor.trainer import PredictorTrainer
from environment.al_env import ALEnvironment
from policy.network import PolicyNetwork, ValueNetwork

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path: str | None) -> Dict[str, Any]:
    default_path = os.path.join(_ROOT, "configs", "default.yaml")
    with open(path or default_path) as f:
        return yaml.safe_load(f)


def _state_to_tensors(state, query_feats, h_obs_dim, h_pool_dim, device):
    ho = torch.tensor(state[:h_obs_dim], dtype=torch.float32).unsqueeze(0).to(device)
    hp = torch.tensor(
        state[h_obs_dim:h_obs_dim + h_pool_dim], dtype=torch.float32
    ).unsqueeze(0).to(device)
    hb = torch.tensor(state[h_obs_dim + h_pool_dim:], dtype=torch.float32).unsqueeze(0).to(device)
    q = torch.tensor(query_feats, dtype=torch.float32).unsqueeze(0).to(device)
    mask = torch.zeros(1, query_feats.shape[0], dtype=torch.bool).to(device)
    return ho, hp, hb, q, mask


# ---------------------------------------------------------------------------
# Single episode runner
# ---------------------------------------------------------------------------

def run_episode(
    env: ALEnvironment,
    policy: PolicyNetwork,
    h_obs_dim: int,
    h_pool_dim: int,
    device: str,
    greedy: bool = True,
    seed_offset: int = 0,
) -> List[Dict[str, Any]]:
    """
    Run one full AL episode and return per-round metrics.

    Parameters
    ----------
    seed_offset : added to config["seed"] so multiple seeds use different D_0.
    """
    # Temporarily shift the seed so D_0 differs across seeds
    orig_seed = env.config.get("seed", 42)
    env.config["seed"] = orig_seed + seed_offset

    state = env.reset()

    env.config["seed"] = orig_seed   # restore

    round_log: List[Dict[str, Any]] = []
    done = False

    while not done:
        query_feats, _ = env.get_candidate_features(state)
        if query_feats.shape[0] == 0:
            break

        ho, hp, hb, q, mask = _state_to_tensors(
            state, query_feats, h_obs_dim, h_pool_dim, device
        )

        with torch.no_grad():
            action_t, _ = policy.act(ho, hp, hb, q, mask, greedy=greedy)
        action = int(action_t.item())

        next_state, reward, done, info = env.step(action)

        # end-of-round: reward != 0 or episode done
        if reward != 0.0 or done:
            round_log.append({
                "round":             info.get("round", -1),
                "num_labeled_cells": info.get("num_labeled_cells", 0),
                "num_labeled_conds": info.get("num_labeled_conds", 0),
                "id_val_mse":        info.get("id_val_mse",  0.0),
                "ood_val_mse":       info.get("ood_val_mse", 0.0),
                "test_mse":          info.get("test_mse",    0.0),
            })
            logger.info(
                f"  round {info.get('round','?')} | "
                f"labeled={info.get('num_labeled_conds','?')} conds | "
                f"ood_mse={info.get('ood_val_mse',0):.4f} | "
                f"test_mse={info.get('test_mse',0):.4f}"
            )

        state = next_state

    return round_log


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fair offline evaluation of trained RL policy"
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to .pt checkpoint (policy_final.pt or checkpoint_iter_XXXX.pt)")
    parser.add_argument("--gene_embeddings", type=str, required=True)
    parser.add_argument("--cell_embeddings", type=str, required=True)
    parser.add_argument("--h5ad", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--greedy", action="store_true", default=True,
                        help="Use greedy (argmax) action selection (default: True)")
    parser.add_argument("--stochastic", dest="greedy", action="store_false",
                        help="Use stochastic action sampling instead of greedy")
    parser.add_argument("--n_seeds", type=int, default=1,
                        help="Number of different D_0 seeds to average over")
    parser.add_argument("--method_name", type=str, default="RL (PPO)",
                        help="Label written into the output CSV")
    parser.add_argument("--output_dir", type=str, default="eval_results")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Config + device
    # ------------------------------------------------------------------
    config = load_config(args.config)

    device_str = args.device or config.get("device", "cpu")
    if device_str == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU.")
        device_str = "cpu"
    device = device_str

    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    logger.info("Loading embeddings...")
    gene_embeddings = sanitize_gene_embeddings(np.load(args.gene_embeddings))
    cell_embeddings = np.load(args.cell_embeddings).astype(np.float32)

    import scanpy as sc
    adata = sc.read_h5ad(args.h5ad)
    _encode_guide_merged(adata)
    num_guides = get_cached_num_guides()
    cell_dim = cell_embeddings.shape[1]

    # ------------------------------------------------------------------
    # Build a FRESH ensemble + trainer (no checkpoint weights)
    # ------------------------------------------------------------------
    logger.info("Building fresh ensemble predictor (no checkpoint weights)...")
    ensemble = EnsemblePredictor(
        gene_embeddings=gene_embeddings,
        cell_dim=cell_dim,
        num_guides=num_guides,
        config=config,
    ).to(device)

    trainer = PredictorTrainer(
        ensemble=ensemble,
        gene_embeddings=gene_embeddings,
        cell_dim=cell_dim,
        num_guides=num_guides,
        config=config,
        device=device,
    )

    # ------------------------------------------------------------------
    # Build environment (same data splits as training)
    # ------------------------------------------------------------------
    logger.info("Building environment...")
    env = ALEnvironment(
        config=config,
        ensemble=ensemble,
        trainer=trainer,
        gene_embeddings=gene_embeddings,
        h5ad_path=args.h5ad,
        cell_embeddings_path=args.cell_embeddings,
        device=device,
    )

    # ------------------------------------------------------------------
    # Load policy weights ONLY
    # ------------------------------------------------------------------
    logger.info(f"Loading policy from {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)

    cfg_s = config["state"]
    cfg_p = config["policy"]
    policy = PolicyNetwork(
        h_obs_dim=cfg_s["h_obs_dim"],
        h_pool_dim=cfg_s["h_pool_dim"],
        h_pb_dim=cfg_s["h_pb_dim"],
        query_dim=cfg_s["candidate_query_dim"],
        attn_dim=cfg_p["cross_attn_dim"],
    ).to(device)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()

    h_obs_dim = cfg_s["h_obs_dim"]
    h_pool_dim = cfg_s["h_pool_dim"]

    # ------------------------------------------------------------------
    # Run evaluation episodes
    # ------------------------------------------------------------------
    all_round_logs: List[Dict[str, Any]] = []

    for seed_i in range(args.n_seeds):
        logger.info(f"=== Eval episode {seed_i + 1}/{args.n_seeds} (seed_offset={seed_i}) ===")

        # Reset predictor to random init before each episode
        logger.info("Resetting predictor weights to random init...")
        ensemble_fresh = EnsemblePredictor(
            gene_embeddings=gene_embeddings,
            cell_dim=cell_dim,
            num_guides=num_guides,
            config=config,
        ).to(device)
        env.ensemble = ensemble_fresh
        env.trainer = PredictorTrainer(
            ensemble=ensemble_fresh,
            gene_embeddings=gene_embeddings,
            cell_dim=cell_dim,
            num_guides=num_guides,
            config=config,
            device=device,
        )

        round_log = run_episode(
            env=env,
            policy=policy,
            h_obs_dim=h_obs_dim,
            h_pool_dim=h_pool_dim,
            device=device,
            greedy=args.greedy,
            seed_offset=seed_i,
        )
        for row in round_log:
            row["seed"] = seed_i
        all_round_logs.extend(round_log)

    # ------------------------------------------------------------------
    # Save CSV (same columns as baseline.py + seed column)
    # ------------------------------------------------------------------
    if not all_round_logs:
        logger.warning("No results collected.")
        return

    out_path = Path(args.output_dir) / "rl_eval.csv"
    fieldnames = ["method", "seed", "round", "num_labeled_cells", "num_labeled_conds",
                  "id_val_mse", "ood_val_mse", "test_mse"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_round_logs:
            writer.writerow({"method": args.method_name, **row})

    logger.info(f"Eval results saved to {out_path}")

    # Quick summary
    last_seed_rows = [r for r in all_round_logs if r["seed"] == args.n_seeds - 1]
    if last_seed_rows:
        final = last_seed_rows[-1]
        logger.info(
            f"Final round — labeled: {final['num_labeled_conds']} conds | "
            f"OOD MSE: {final['ood_val_mse']:.4f} | Test MSE: {final['test_mse']:.4f}"
        )


if __name__ == "__main__":
    main()
