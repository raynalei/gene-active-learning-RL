"""
main.py — Full three-stage active learning training pipeline.

Stage 1: Teacher rollout generation (heuristic policy)
Stage 2: Behaviour cloning warm-start
Stage 3: PPO fine-tuning with Dyna simulation

Usage
-----
python main.py \
    --gene_embeddings path/to/gene_embs.npy \
    --cell_embeddings path/to/cell_embs.npy \
    --h5ad            path/to/norman2019.h5ad \
    [--config         path/to/config.yaml]    \
    [--output_dir     results/]               \
    [--device         cuda]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any, Dict

import numpy as np
import torch
import yaml

# Ensure project root is on path
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from predict import sanitize_gene_embeddings, get_cached_num_guides, _encode_guide_merged
from predictor.ensemble import EnsemblePredictor
from predictor.trainer import PredictorTrainer
from environment.al_env import ALEnvironment
from policy.network import PolicyNetwork, ValueNetwork
from simulator.batch_simulator import BatchSimulator
from training.bc_warmstart import generate_teacher_rollouts, train_bc
from training.ppo_trainer import PPOTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: str | None) -> Dict[str, Any]:
    default_path = os.path.join(_ROOT, "configs", "default.yaml")
    cfg_path = path or default_path
    with open(cfg_path) as f:
        config = yaml.safe_load(f)
    return config


# ---------------------------------------------------------------------------
# Build model objects
# ---------------------------------------------------------------------------

def build_ensemble(
    gene_embeddings: np.ndarray,
    cell_dim: int,
    num_guides: int,
    config: Dict[str, Any],
    device: str,
) -> EnsemblePredictor:
    ensemble = EnsemblePredictor(
        gene_embeddings=gene_embeddings,
        cell_dim=cell_dim,
        num_guides=num_guides,
        config=config,
    ).to(device)
    return ensemble


def build_policy(config: Dict[str, Any]) -> tuple[PolicyNetwork, ValueNetwork]:
    cfg_s = config["state"]
    cfg_p = config["policy"]
    policy = PolicyNetwork(
        h_obs_dim=cfg_s["h_obs_dim"],
        h_pool_dim=cfg_s["h_pool_dim"],
        h_pb_dim=cfg_s["h_pb_dim"],
        query_dim=cfg_s["candidate_query_dim"],
        attn_dim=cfg_p["cross_attn_dim"],
    )
    value_net = ValueNetwork(
        state_dim=cfg_s["state_dim"],
        hidden_dim=cfg_p["value_hidden_dim"],
    )
    return policy, value_net


def build_simulator(
    config: Dict[str, Any],
    state_dim: int,
    phi_dim: int,
    device: str,
) -> BatchSimulator:
    return BatchSimulator(
        config=config,
        state_dim=state_dim,
        phi_dim=phi_dim,
        device=device,
    )


# ---------------------------------------------------------------------------
# phi_dim calculation
# ---------------------------------------------------------------------------

def compute_phi_dim(config: Dict[str, Any]) -> int:
    """phi(x) = state + z_x + u(x) + d_Dt + d_B"""
    cfg_s = config["state"]
    return cfg_s["state_dim"] + cfg_s["embedding_dim"] + 3


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    config = load_config(args.config)

    # --seed overrides config
    if args.seed is not None:
        config["seed"] = args.seed

    # --override key.subkey=value  (e.g. reward.w_cov=0)
    for kv in args.override:
        key_path, _, value_str = kv.partition("=")
        keys = key_path.strip().split(".")
        value = yaml.safe_load(value_str)
        d = config
        for k in keys[:-1]:
            d = d[k]
        d[keys[-1]] = value
        logger.info(f"Config override: {key_path} = {value}")

    # --no_dyna: zero out Dyna ratios before PPOTrainer is built
    if args.no_dyna:
        config["ppo"]["dyna_ratio_early"] = 0
        config["ppo"]["dyna_ratio_late"] = 0
        logger.info("Dyna disabled (--no_dyna)")

    # Override device from args if provided
    device_str = args.device or config.get("device", "cpu")
    if device_str == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU.")
        device_str = "cpu"
    device = device_str

    torch.manual_seed(config.get("seed", 42))
    np.random.seed(config.get("seed", 42))

    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    logger.info("Loading gene and cell embeddings...")
    gene_embeddings = sanitize_gene_embeddings(np.load(args.gene_embeddings))
    cell_embeddings = np.load(args.cell_embeddings).astype(np.float32)

    # Trigger guide encoding so num_guides is populated
    import scanpy as sc
    adata = sc.read_h5ad(args.h5ad)
    _encode_guide_merged(adata)
    num_guides = get_cached_num_guides()

    cell_dim = cell_embeddings.shape[1]
    logger.info(f"gene_embeddings: {gene_embeddings.shape}, cell_dim: {cell_dim}, num_guides: {num_guides}")

    # ------------------------------------------------------------------
    # Build components
    # ------------------------------------------------------------------
    logger.info("Building ensemble predictor...")
    ensemble = build_ensemble(gene_embeddings, cell_dim, num_guides, config, device)

    trainer = PredictorTrainer(
        ensemble=ensemble,
        gene_embeddings=gene_embeddings,
        cell_dim=cell_dim,
        num_guides=num_guides,
        config=config,
        device=device,
    )

    logger.info("Building active learning environment...")
    env = ALEnvironment(
        config=config,
        ensemble=ensemble,
        trainer=trainer,
        gene_embeddings=gene_embeddings,
        h5ad_path=args.h5ad,
        cell_embeddings_path=args.cell_embeddings,
        device=device,
    )

    policy, value_net = build_policy(config)

    phi_dim = compute_phi_dim(config)
    state_dim = config["state"]["state_dim"]
    simulator = build_simulator(config, state_dim, phi_dim, device)

    # ------------------------------------------------------------------
    # Stage 1 + 2: Teacher rollouts → Behaviour Cloning
    # ------------------------------------------------------------------
    if not args.no_bc:
        logger.info("=" * 60)
        logger.info("Stage 1: Generating teacher rollouts...")
        teacher_transitions = generate_teacher_rollouts(env, config)

        logger.info("=" * 60)
        logger.info("Stage 2: Behaviour cloning warm-start...")
        policy = train_bc(policy, teacher_transitions, config, device)

        bc_path = os.path.join(args.output_dir, "policy_bc.pt")
        torch.save({
            "policy": policy.state_dict(),
            "value_net": value_net.state_dict(),
        }, bc_path)
        logger.info(f"BC checkpoint saved to {bc_path}")
    else:
        logger.info("Skipping BC warm-start (--no_bc)")

    # ------------------------------------------------------------------
    # Stage 3: PPO fine-tuning with Dyna
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Stage 3: PPO fine-tuning with Dyna...")

    n_ppo_iters = config["active_learning"]["num_rounds"] * 5  # configurable

    ppo_trainer = PPOTrainer(
        env=env,
        policy=policy,
        value_net=value_net,
        simulator=simulator,
        config=config,
        device=device,
    )

    logs, round_log = ppo_trainer.train(n_ppo_iters)

    # ------------------------------------------------------------------
    # Save final checkpoint and logs
    # ------------------------------------------------------------------
    final_path = os.path.join(args.output_dir, "policy_final.pt")
    torch.save({
        "policy": policy.state_dict(),
        "value_net": value_net.state_dict(),
        "config": config,
    }, final_path)
    logger.info(f"Final policy saved to {final_path}")

    import csv

    # Per-PPO-iteration log
    log_path = os.path.join(args.output_dir, "training_log.csv")
    if logs:
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=logs[0].keys())
            writer.writeheader()
            writer.writerows(logs)
    logger.info(f"Training log saved to {log_path}")

    # Per-AL-round log — directly comparable to random_sample.py's CSV output
    round_log_path = os.path.join(args.output_dir, "round_log.csv")
    if round_log:
        with open(round_log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=round_log[0].keys())
            writer.writeheader()
            writer.writerows(round_log)
    logger.info(f"Round-level log saved to {round_log_path}")

    # Summary
    ood_curve = [l["ood_val_mse"] for l in logs]
    test_curve = [l["test_mse"] for l in logs]
    logger.info(f"OOD MSE:  start={ood_curve[0]:.4f}, end={ood_curve[-1]:.4f}, min={min(ood_curve):.4f}")
    logger.info(f"Test MSE: start={test_curve[0]:.4f}, end={test_curve[-1]:.4f}, min={min(test_curve):.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gene AL-RL training pipeline")
    parser.add_argument("--gene_embeddings", type=str, required=True,
                        help="Path to gene embedding .npy [G, Dg]")
    parser.add_argument("--cell_embeddings", type=str, required=True,
                        help="Path to cell embedding .npy [N, Dc]")
    parser.add_argument("--h5ad", type=str, required=True,
                        help="Path to Norman2019 .h5ad file")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config YAML (default: configs/default.yaml)")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory for checkpoints and logs")
    parser.add_argument("--device", type=str, default=None,
                        help="torch device: 'cuda' or 'cpu' (overrides config)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (overrides config seed)")
    parser.add_argument("--override", type=str, nargs="*", default=[],
                        help="Override config values: section.key=value "
                             "(e.g. reward.w_cov=0 ppo.gamma=0.99)")
    parser.add_argument("--no_bc", action="store_true",
                        help="Skip Stage 2 behaviour cloning warm-start")
    parser.add_argument("--no_dyna", action="store_true",
                        help="Disable Stage 3 Dyna model-based rollout augmentation")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
