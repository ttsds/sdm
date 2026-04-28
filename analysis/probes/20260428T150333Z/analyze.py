"""Analyse the probe matrix produced by scripts/run_linear_probes.py.

Run from anywhere; outputs land in this directory:

    .venv/bin/python analysis/probes/20260428T150333Z/analyze.py

Inputs (next to this script):
  - matrix.json   8 × 8 × 25 grid of (sdm, target, layer) → R² test/train
  - manifest.json experiment metadata staged by consolidate_weights.py

Outputs:
  - summary.txt              text report
  - best_layer_matrix.csv    R²_test at each row's best layer
  - best_layer_indices.csv   the layer index that achieved that R²
  - r2_full_long.csv         long-form CSV of every (sdm, target, layer)
  - heatmap_best_layer.png   per-row best-layer R²_test heatmap
  - layer_trajectory_diag.png R²_test vs layer for sdm-X → sdm-X diagonals
  - factor_block.png         factor-averaged R² heatmap (4 × 4)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

# --- Factor taxonomy (matches ARCHITECTURE.md, sans the excluded wespeaker run) ---
FACTOR_OF = {
    "sdm-xlsr-fairseq": "Generic",
    "sdm-dvector":      "Speaker",
    "sdm-pitch":        "Prosody",
    "sdm-mpm":          "Prosody",
    "sdm-speaking-rate":"Prosody",
    "sdm-w2v2-asr":     "Intelligibility",
    "sdm-mwhisper":     "Intelligibility",
    "sdm-emotion2vec":  "Emotion",
}

FACTOR_ORDER = ["Generic", "Speaker", "Prosody", "Intelligibility", "Emotion"]


def _ordered_experiments(experiments: set[str]) -> list[str]:
    """Stable order: Generic → Speaker → Prosody → Intelligibility → Emotion."""
    keyed: list[tuple[int, str]] = []
    for exp in experiments:
        factor = FACTOR_OF.get(exp, "Other")
        keyed.append((FACTOR_ORDER.index(factor) if factor in FACTOR_ORDER else 999, exp))
    return [e for _, e in sorted(keyed)]


def load_matrix(path: Path) -> tuple[np.ndarray, np.ndarray, list[str], list[int]]:
    """Return (R²_test, R²_train, experiments, layers).

    Both R² arrays are shaped (n_sdm, n_target, n_layers).
    """
    rows = json.loads(path.read_text())
    df = pd.DataFrame(rows)
    experiments = _ordered_experiments(set(df["sdm"]) | set(df["target"]))
    layers = sorted(df["layer"].unique())
    n_e, n_l = len(experiments), len(layers)
    r2_test = np.full((n_e, n_e, n_l), np.nan, dtype=np.float64)
    r2_train = np.full_like(r2_test, np.nan)
    e_idx = {e: i for i, e in enumerate(experiments)}
    l_idx = {layer: i for i, layer in enumerate(layers)}
    for r in rows:
        i = e_idx[r["sdm"]]
        j = e_idx[r["target"]]
        k = l_idx[r["layer"]]
        r2_test[i, j, k] = r["r2_test"]
        r2_train[i, j, k] = r["r2_train"]
    return r2_test, r2_train, experiments, layers


def best_layer_per_pair(r2: np.ndarray, layers: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Return (best_r2, best_layer) shaped (n_sdm, n_target)."""
    # NaN-safe argmax (ignore NaN; if a row is all-NaN, return -1).
    masked = np.where(np.isnan(r2), -np.inf, r2)
    best_idx = masked.argmax(axis=-1)
    best_r2 = np.take_along_axis(r2, best_idx[..., None], axis=-1).squeeze(-1)
    best_layer = np.array(layers)[best_idx]
    return best_r2, best_layer


def factor_block_average(r2_at_best: np.ndarray, experiments: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Average best-layer R² within each (factor_sdm, factor_target) cell."""
    factors_present = [f for f in FACTOR_ORDER if any(FACTOR_OF.get(e) == f for e in experiments)]
    df = pd.DataFrame(r2_at_best, index=experiments, columns=experiments)
    long = df.stack().reset_index()
    long.columns = ["sdm", "target", "r2"]
    long["sdm_factor"] = long["sdm"].map(FACTOR_OF)
    long["target_factor"] = long["target"].map(FACTOR_OF)
    block = (
        long.groupby(["sdm_factor", "target_factor"])["r2"]
        .mean()
        .unstack("target_factor")
        .reindex(index=factors_present, columns=factors_present)
    )
    return block, factors_present


def write_csvs(out: Path, experiments: list[str], layers: list[int],
               r2_test: np.ndarray, best_r2: np.ndarray, best_layer: np.ndarray) -> None:
    pd.DataFrame(best_r2, index=experiments, columns=experiments).to_csv(out / "best_layer_matrix.csv")
    pd.DataFrame(best_layer, index=experiments, columns=experiments).to_csv(out / "best_layer_indices.csv")
    long_rows = []
    for i, sdm in enumerate(experiments):
        for j, tgt in enumerate(experiments):
            for k, layer in enumerate(layers):
                long_rows.append(
                    {"sdm": sdm, "target": tgt, "layer": layer, "r2_test": r2_test[i, j, k]}
                )
    pd.DataFrame(long_rows).to_csv(out / "r2_full_long.csv", index=False)


def plot_best_layer_heatmap(
    out: Path, experiments: list[str], best_r2: np.ndarray, best_layer: np.ndarray
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = plt.get_cmap("RdBu_r")
    im = ax.imshow(best_r2, cmap=cmap, vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(len(experiments)))
    ax.set_xticklabels(experiments, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(experiments)))
    ax.set_yticklabels(experiments, fontsize=8)
    ax.set_xlabel("teacher target")
    ax.set_ylabel("backbone (SDM_i)")
    for i in range(best_r2.shape[0]):
        for j in range(best_r2.shape[1]):
            value = best_r2[i, j]
            layer = int(best_layer[i, j])
            color = "white" if abs(value) > 0.5 else "black"
            ax.text(j, i, f"{value:+.2f}\nL{layer}", ha="center", va="center", fontsize=7, color=color)
    fig.colorbar(im, ax=ax, label="R² test (best layer)")
    ax.set_title("Cross-teacher probe matrix\nbest layer per (backbone, target)")
    fig.tight_layout()
    fig.savefig(out / "heatmap_best_layer.png", dpi=160)
    plt.close(fig)


def plot_layer_trajectories_diag(
    out: Path, experiments: list[str], layers: list[int], r2_test: np.ndarray
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap("tab10")
    for idx, exp in enumerate(experiments):
        i = experiments.index(exp)
        trajectory = r2_test[i, i, :]
        ax.plot(layers, trajectory, marker="o", markersize=3, lw=1.5,
                color=cmap(idx % 10), label=exp)
    ax.axhline(0.0, color="grey", lw=0.5, ls="--")
    ax.set_xlabel("backbone layer")
    ax.set_ylabel("R² test (self-prediction)")
    ax.set_title("Per-layer R² of each SDM_i predicting its own teacher")
    ax.legend(fontsize=7, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.25))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "layer_trajectory_diag.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_factor_block(out: Path, block: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    cmap = plt.get_cmap("RdBu_r")
    im = ax.imshow(block.values, cmap=cmap, vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(len(block.columns)))
    ax.set_xticklabels(block.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(block.index)))
    ax.set_yticklabels(block.index)
    ax.set_xlabel("target factor")
    ax.set_ylabel("backbone factor")
    for i in range(block.shape[0]):
        for j in range(block.shape[1]):
            value = block.values[i, j]
            color = "white" if abs(value) > 0.5 else "black"
            ax.text(j, i, f"{value:+.2f}", ha="center", va="center", fontsize=8, color=color)
    fig.colorbar(im, ax=ax, label="R² test (avg over best-layer)")
    ax.set_title("Factor-averaged probe matrix")
    fig.tight_layout()
    fig.savefig(out / "factor_block.png", dpi=160)
    plt.close(fig)


def write_summary(
    out: Path,
    experiments: list[str],
    layers: list[int],
    r2_test: np.ndarray,
    best_r2: np.ndarray,
    best_layer: np.ndarray,
    block: pd.DataFrame,
) -> None:
    lines: list[str] = []
    lines.append(f"# Probe matrix summary  ({len(experiments)} experiments × {len(layers)} layers)")
    lines.append("")
    lines.append("Diagonal (self-prediction): each SDM_i predicting its own teacher target")
    lines.append("at the best layer.")
    lines.append("")
    lines.append("| backbone | best layer | best R²_test |")
    lines.append("|---|---:|---:|")
    for i, exp in enumerate(experiments):
        bl = int(best_layer[i, i])
        br = best_r2[i, i]
        lines.append(f"| {exp} | {bl} | {br:+.3f} |")
    lines.append("")

    lines.append("Off-diagonal: top 5 strongest cross-teacher predictions (R²_test, best layer).")
    rows = []
    for i, sdm in enumerate(experiments):
        for j, tgt in enumerate(experiments):
            if i == j:
                continue
            rows.append((best_r2[i, j], int(best_layer[i, j]), sdm, tgt))
    rows.sort(reverse=True)
    for r2, layer, sdm, tgt in rows[:5]:
        lines.append(f"  {sdm:>20s} → {tgt:<20s}  R²={r2:+.3f}  L{layer}")
    lines.append("")

    lines.append("Bottom 5 weakest cross-teacher predictions:")
    for r2, layer, sdm, tgt in rows[-5:]:
        lines.append(f"  {sdm:>20s} → {tgt:<20s}  R²={r2:+.3f}  L{layer}")
    lines.append("")

    lines.append("Factor block (mean R² over best-layer cells, grouped by factor):")
    lines.append("")
    lines.append(block.round(3).to_string())
    lines.append("")

    lines.append("Per-row argmax-layer histogram (where each backbone peaks across all targets):")
    flat_layers = best_layer.ravel()
    for layer in sorted(set(flat_layers.tolist())):
        count = int((flat_layers == layer).sum())
        bar = "#" * count
        lines.append(f"  L{layer:>2}: {count:>3}  {bar}")
    lines.append("")

    (out / "summary.txt").write_text("\n".join(lines) + "\n")


def main() -> None:
    matrix_path = HERE / "matrix.json"
    if not matrix_path.exists():
        raise SystemExit(f"missing {matrix_path}")

    r2_test, _r2_train, experiments, layers = load_matrix(matrix_path)
    best_r2, best_layer = best_layer_per_pair(r2_test, layers)
    block, _factors = factor_block_average(best_r2, experiments)

    write_csvs(HERE, experiments, layers, r2_test, best_r2, best_layer)
    plot_best_layer_heatmap(HERE, experiments, best_r2, best_layer)
    plot_layer_trajectories_diag(HERE, experiments, layers, r2_test)
    plot_factor_block(HERE, block)
    write_summary(HERE, experiments, layers, r2_test, best_r2, best_layer, block)

    print("[done] outputs in", HERE)
    for path in sorted(HERE.iterdir()):
        if path.name == "analyze.py":
            continue
        print(" ", path.name, f"({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
