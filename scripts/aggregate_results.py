"""
Aggregate multi-seed results into mean ± std curves.

Reads:
  {output_base}/rl/seed{N}/round_log.csv       — RL per-round metrics
  {output_base}/random/seed{N}/random_al_curve.csv  — Random per-round metrics

  Optionally also reads ablation results under:
  {output_base}/ablation/{variant}/seed{N}/round_log.csv

Outputs:
  {output_base}/aggregated/rl_agg.csv
  {output_base}/aggregated/random_agg.csv
  {output_base}/aggregated/{variant}_agg.csv   (one per ablation)
  {output_base}/aggregated/comparison_test_mse.png
  {output_base}/aggregated/comparison_ood_mse.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _find_seed_dirs(base: Path) -> List[Path]:
    """Return sorted list of seed* subdirectories under *base*."""
    return sorted(p for p in base.iterdir() if p.is_dir() and p.name.startswith("seed"))


def _load_rl_seeds(rl_base: Path) -> Optional[pd.DataFrame]:
    """Load and concatenate round_log.csv files from all RL seed dirs."""
    dfs = []
    for seed_dir in _find_seed_dirs(rl_base):
        csv = seed_dir / "round_log.csv"
        if not csv.exists():
            print(f"  [warn] missing: {csv}")
            continue
        df = pd.read_csv(csv)
        df["seed"] = seed_dir.name
        dfs.append(df)
    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)


def _load_random_seeds(random_base: Path) -> Optional[pd.DataFrame]:
    """Load and concatenate random_al_curve.csv files from all seed dirs."""
    dfs = []
    for seed_dir in _find_seed_dirs(random_base):
        csv = seed_dir / "random_al_curve.csv"
        if not csv.exists():
            print(f"  [warn] missing: {csv}")
            continue
        df = pd.read_csv(csv)
        # Rename to match RL columns
        df = df.rename(columns={
            "num_labeled": "num_labeled_cells",
            "best_val_mse": "id_val_mse",
        })
        df["seed"] = seed_dir.name
        dfs.append(df)
    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _aggregate(df: pd.DataFrame, x_col: str, y_cols: List[str]) -> pd.DataFrame:
    """
    Group by x_col, compute mean and std for each y_col across seeds.

    Returns DataFrame with columns: [x_col, {y}_mean, {y}_std, seed_count].
    """
    grouped = df.groupby(x_col)[y_cols]
    mean = grouped.mean().add_suffix("_mean")
    std = grouped.std(ddof=1).fillna(0).add_suffix("_std")
    count = grouped.size().rename("seed_count")
    return pd.concat([mean, std, count], axis=1).reset_index()


def aggregate_method(df: Optional[pd.DataFrame], x_col: str, y_cols: List[str]) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None
    available = [c for c in y_cols if c in df.columns]
    if not available:
        return None
    return _aggregate(df, x_col, available)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

COLORS = {
    "rl": "#d62728",
    "random": "#1f77b4",
    "no_bc": "#ff7f0e",
    "no_dyna": "#2ca02c",
    "no_bc_no_dyna": "#9467bd",
    "no_w_cov": "#8c564b",
    "no_w_unc": "#e377c2",
    "no_w_des": "#7f7f7f",
    "no_w_red": "#bcbd22",
}


def _plot_curve(
    agg_dict: Dict[str, pd.DataFrame],
    x_col: str,
    y_col: str,
    ylabel: str,
    title: str,
    save_path: Path,
) -> None:
    plt.figure(figsize=(8, 5))
    for name, agg in agg_dict.items():
        if agg is None:
            continue
        mean_col = f"{y_col}_mean"
        std_col = f"{y_col}_std"
        if mean_col not in agg.columns:
            continue
        x = agg[x_col].values
        y_mean = agg[mean_col].values
        y_std = agg[std_col].values if std_col in agg.columns else np.zeros_like(y_mean)
        color = COLORS.get(name, None)
        label = name.replace("_", " ")
        plt.plot(x, y_mean, marker="o", markersize=4, linewidth=2,
                 label=label, color=color)
        plt.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.15, color=color)

    plt.xlabel("Number of labeled cells")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate multi-seed AL results")
    parser.add_argument("--output_base", type=str, default="results",
                        help="Root results directory (default: results)")
    parser.add_argument("--ablation_base", type=str, default=None,
                        help="Ablation results dir (default: {output_base}/ablation)")
    args = parser.parse_args()

    base = Path(args.output_base)
    abl_base = Path(args.ablation_base) if args.ablation_base else base / "ablation"
    out_dir = base / "aggregated"
    out_dir.mkdir(parents=True, exist_ok=True)

    X_COL = "num_labeled_cells"
    Y_COLS = ["test_mse", "ood_val_mse", "id_val_mse"]

    # ------------------------------------------------------------------
    # Load RL + Random
    # ------------------------------------------------------------------
    print("Loading RL results...")
    rl_raw = _load_rl_seeds(base / "rl") if (base / "rl").exists() else None
    rl_agg = aggregate_method(rl_raw, X_COL, Y_COLS)

    print("Loading Random results...")
    rand_raw = _load_random_seeds(base / "random") if (base / "random").exists() else None
    rand_agg = aggregate_method(rand_raw, X_COL, Y_COLS)

    if rl_agg is not None:
        rl_agg.to_csv(out_dir / "rl_agg.csv", index=False)
        print(f"  Saved: {out_dir / 'rl_agg.csv'}")
    if rand_agg is not None:
        rand_agg.to_csv(out_dir / "random_agg.csv", index=False)
        print(f"  Saved: {out_dir / 'random_agg.csv'}")

    # ------------------------------------------------------------------
    # Load ablations
    # ------------------------------------------------------------------
    ablation_variants = [
        "no_w_cov", "no_w_unc", "no_w_des", "no_w_red",
        "no_bc", "no_dyna", "no_bc_no_dyna",
    ]
    abl_aggs: Dict[str, Optional[pd.DataFrame]] = {}

    for variant in ablation_variants:
        variant_dir = abl_base / variant
        if not variant_dir.exists():
            continue
        print(f"Loading ablation: {variant}...")
        raw = _load_rl_seeds(variant_dir)
        agg = aggregate_method(raw, X_COL, Y_COLS)
        if agg is not None:
            abl_aggs[variant] = agg
            agg.to_csv(out_dir / f"{variant}_agg.csv", index=False)
            print(f"  Saved: {out_dir / f'{variant}_agg.csv'}")

    # ------------------------------------------------------------------
    # Comparison plots: RL vs Random (+ ablations if available)
    # ------------------------------------------------------------------
    main_dict: Dict[str, Optional[pd.DataFrame]] = {"rl": rl_agg, "random": rand_agg}

    for y_col, ylabel, title_suffix in [
        ("test_mse",    "Test MSE",    "Test MSE"),
        ("ood_val_mse", "OOD Val MSE", "OOD Val MSE"),
    ]:
        _plot_curve(
            main_dict, X_COL, y_col, ylabel,
            f"RL vs Random — {title_suffix}",
            out_dir / f"comparison_{y_col}.png",
        )

    # ------------------------------------------------------------------
    # Ablation plots (RL full vs each ablation variant)
    # ------------------------------------------------------------------
    if abl_aggs:
        for y_col, ylabel in [("test_mse", "Test MSE"), ("ood_val_mse", "OOD Val MSE")]:
            abl_plot_dict: Dict[str, Optional[pd.DataFrame]] = {"rl (full)": rl_agg}
            abl_plot_dict.update(abl_aggs)
            _plot_curve(
                abl_plot_dict, X_COL, y_col, ylabel,
                f"Ablation — {ylabel}",
                out_dir / f"ablation_{y_col}.png",
            )

    print(f"\nDone. All aggregated results in: {out_dir}")


if __name__ == "__main__":
    main()
