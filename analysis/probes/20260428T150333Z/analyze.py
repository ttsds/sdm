"""Analyse the probe matrix produced by scripts/run_linear_probes.py.

Run from anywhere; outputs land in this directory:

    .venv/bin/python analysis/probes/20260428T150333Z/analyze.py

Inputs (next to this script):
  - matrix.json   grid of (sdm, target, layer[, probe_type]) → R² test/train
  - manifest.json experiment metadata staged by consolidate_weights.py

Outputs:
  - summary.txt                   text report (all probe types)
  - r2_full_long.csv              long-form CSV of every row (all probe types)
  - best_layer_matrix_<pt>.csv    R²_test at each row's best layer per probe type
  - best_layer_indices_<pt>.csv   layer index that achieved that R²
  - heatmap_best_layer_<pt>.png   per-row best-layer R²_test heatmap
  - layer_trajectory_diag_<pt>.png  R²_test vs layer for diagonal entries
  - factor_block_<pt>.png         factor-averaged R² heatmap
  - heatmap_comparison.png        side-by-side ridge vs MLP (if both present)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

FACTOR_OF = {
    "sdm-xlsr-fairseq": "Generic",
    "sdm-xlsr":         "Generic",
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
    keyed = [(FACTOR_ORDER.index(FACTOR_OF.get(e, "Other"))
              if FACTOR_OF.get(e) in FACTOR_ORDER else 999, e)
             for e in experiments]
    return [e for _, e in sorted(keyed)]


def available_probe_types(rows: list[dict]) -> list[str]:
    """Stable order: ridge before mlp."""
    seen = {r.get("probe_type", "ridge") for r in rows}
    return [pt for pt in ("ridge", "mlp") if pt in seen]


def load_matrix(rows: list[dict], probe_type: str) -> tuple[np.ndarray, np.ndarray, list[str], list[int]]:
    """Filter to one probe type; return (R²_test, R²_train, experiments, layers).

    Old matrices without a ``probe_type`` field are treated as 'ridge'.
    Arrays are shaped (n_sdm, n_target, n_layers).
    """
    filtered = [r for r in rows if r.get("probe_type", "ridge") == probe_type]
    df = pd.DataFrame(filtered)
    experiments = _ordered_experiments(set(df["sdm"]) | set(df["target"]))
    layers = sorted(df["layer"].unique())
    n_e, n_l = len(experiments), len(layers)
    r2_test = np.full((n_e, n_e, n_l), np.nan, dtype=np.float64)
    r2_train = np.full_like(r2_test, np.nan)
    e_idx = {e: i for i, e in enumerate(experiments)}
    l_idx = {layer: i for i, layer in enumerate(layers)}
    for r in filtered:
        i = e_idx.get(r["sdm"])
        j = e_idx.get(r["target"])
        k = l_idx.get(r["layer"])
        if i is None or j is None or k is None:
            continue
        r2_test[i, j, k] = r["r2_test"]
        r2_train[i, j, k] = r["r2_train"]
    return r2_test, r2_train, experiments, layers


def best_layer_per_pair(r2: np.ndarray, layers: list[int]) -> tuple[np.ndarray, np.ndarray]:
    masked = np.where(np.isnan(r2), -np.inf, r2)
    best_idx = masked.argmax(axis=-1)
    best_r2 = np.take_along_axis(r2, best_idx[..., None], axis=-1).squeeze(-1)
    best_layer = np.array(layers)[best_idx]
    return best_r2, best_layer


def factor_block_average(best_r2: np.ndarray, experiments: list[str]) -> pd.DataFrame:
    factors_present = [f for f in FACTOR_ORDER
                       if any(FACTOR_OF.get(e) == f for e in experiments)]
    df = pd.DataFrame(best_r2, index=experiments, columns=experiments)
    long = df.stack().reset_index()
    long.columns = ["sdm", "target", "r2"]
    long["sdm_factor"] = long["sdm"].map(FACTOR_OF)
    long["target_factor"] = long["target"].map(FACTOR_OF)
    return (long.groupby(["sdm_factor", "target_factor"])["r2"]
            .mean().unstack("target_factor")
            .reindex(index=factors_present, columns=factors_present))


def _heatmap_ax(ax, data: np.ndarray, row_labels: list[str], col_labels: list[str],
                cell_text=None, vmin=-1.0, vmax=1.0, cmap="RdBu_r", title=""):
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=40, ha="right", fontsize=7)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=7)
    if cell_text is not None:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                v = data[i, j]
                color = "white" if abs(v) > 0.5 else "black"
                ax.text(j, i, cell_text[i, j], ha="center", va="center",
                        fontsize=6, color=color)
    ax.set_title(title, fontsize=9)
    return im


def plot_best_layer_heatmap(out: Path, probe_type: str, experiments: list[str],
                             best_r2: np.ndarray, best_layer: np.ndarray) -> None:
    cell = np.empty(best_r2.shape, dtype=object)
    for i in range(best_r2.shape[0]):
        for j in range(best_r2.shape[1]):
            cell[i, j] = f"{best_r2[i,j]:+.2f}\nL{int(best_layer[i,j])}"
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = _heatmap_ax(ax, best_r2, experiments, experiments, cell_text=cell,
                     title=f"Cross-teacher probe matrix ({probe_type}, best layer per row)")
    ax.set_xlabel("teacher target"); ax.set_ylabel("backbone (SDM_i)")
    fig.colorbar(im, ax=ax, label="R² test")
    fig.tight_layout()
    fig.savefig(out / f"heatmap_best_layer_{probe_type}.png", dpi=160)
    plt.close(fig)


def plot_layer_trajectories_diag(out: Path, probe_type: str, experiments: list[str],
                                  layers: list[int], r2_test: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap("tab10")
    for idx, exp in enumerate(experiments):
        i = experiments.index(exp)
        ax.plot(layers, r2_test[i, i, :], marker="o", markersize=3, lw=1.5,
                color=cmap(idx % 10), label=exp)
    ax.axhline(0.0, color="grey", lw=0.5, ls="--")
    ax.set_xlabel("backbone layer"); ax.set_ylabel("R² test (self-prediction)")
    ax.set_title(f"Per-layer R² — each SDM_i predicting its own teacher ({probe_type})")
    ax.legend(fontsize=7, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.25))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / f"layer_trajectory_diag_{probe_type}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_factor_block(out: Path, probe_type: str, block: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = _heatmap_ax(ax, block.values,
                     list(block.index), list(block.columns),
                     cell_text=np.array([[f"{v:+.2f}" for v in row] for row in block.values]),
                     title=f"Factor-averaged R² ({probe_type})")
    ax.set_xlabel("target factor"); ax.set_ylabel("backbone factor")
    fig.colorbar(im, ax=ax, label="R² test (avg over best-layer)")
    fig.tight_layout()
    fig.savefig(out / f"factor_block_{probe_type}.png", dpi=160)
    plt.close(fig)


def plot_comparison(out: Path, data: dict) -> None:
    """Side-by-side best-layer heatmap for every probe type that's present."""
    probe_types = list(data)
    n = len(probe_types)
    if n < 2:
        return
    fig, axes = plt.subplots(1, n, figsize=(7.5 * n, 6.5), squeeze=False)
    axes = axes[0]
    for ax, pt in zip(axes, probe_types):
        best_r2, best_layer, experiments = data[pt]["best_r2"], data[pt]["best_layer"], data[pt]["experiments"]
        cell = np.empty(best_r2.shape, dtype=object)
        for i in range(best_r2.shape[0]):
            for j in range(best_r2.shape[1]):
                cell[i, j] = f"{best_r2[i,j]:+.2f}\nL{int(best_layer[i,j])}"
        im = _heatmap_ax(ax, best_r2, experiments, experiments, cell_text=cell,
                         title=f"{pt} — best-layer R²_test")
        ax.set_xlabel("teacher target")
        ax.set_ylabel("backbone")
        fig.colorbar(im, ax=ax, label="R² test")
    fig.suptitle("Ridge vs MLP probes — best layer per (backbone, target)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "heatmap_comparison.png", dpi=160)
    plt.close(fig)


def write_csvs(out: Path, probe_type: str, experiments: list[str], layers: list[int],
               r2_test: np.ndarray, best_r2: np.ndarray, best_layer: np.ndarray) -> None:
    pd.DataFrame(best_r2, index=experiments, columns=experiments).to_csv(
        out / f"best_layer_matrix_{probe_type}.csv")
    pd.DataFrame(best_layer, index=experiments, columns=experiments).to_csv(
        out / f"best_layer_indices_{probe_type}.csv")


def write_summary(out: Path, all_data: dict) -> None:
    lines: list[str] = ["# Probe matrix summary\n"]
    for probe_type, d in all_data.items():
        experiments = d["experiments"]
        best_r2 = d["best_r2"]
        best_layer = d["best_layer"]
        block = d["block"]
        layers = d["layers"]
        lines.append(f"## {probe_type}  ({len(experiments)} experiments × {len(layers)} layers)\n")
        lines.append("Diagonal (self-prediction):\n")
        lines.append("| backbone | best layer | R²_test |")
        lines.append("|---|---:|---:|")
        for i, exp in enumerate(experiments):
            lines.append(f"| {exp} | {int(best_layer[i,i])} | {best_r2[i,i]:+.3f} |")
        lines.append("")
        rows = [(best_r2[i,j], int(best_layer[i,j]), sdm, tgt)
                for i, sdm in enumerate(experiments)
                for j, tgt in enumerate(experiments) if i != j]
        rows.sort(reverse=True)
        lines.append("Top 5 cross-teacher predictions:")
        for r2, layer, sdm, tgt in rows[:5]:
            lines.append(f"  {sdm:>20s} → {tgt:<20s}  R²={r2:+.3f}  L{layer}")
        lines.append("")
        lines.append("Factor block:")
        lines.append(block.round(3).to_string())
        lines.append("\n")
    (out / "summary.txt").write_text("\n".join(lines) + "\n")


def main() -> None:
    matrix_path = HERE / "matrix.json"
    if not matrix_path.exists():
        raise SystemExit(f"missing {matrix_path}")

    rows = json.loads(matrix_path.read_text())
    probe_types = available_probe_types(rows)
    print(f"[analyze] probe types in matrix: {probe_types}")

    # Long CSV covering all probe types
    pd.DataFrame(rows).to_csv(HERE / "r2_full_long.csv", index=False)

    all_data: dict = {}
    for probe_type in probe_types:
        r2_test, _r2_train, experiments, layers = load_matrix(rows, probe_type)
        best_r2, best_layer = best_layer_per_pair(r2_test, layers)
        block = factor_block_average(best_r2, experiments)
        all_data[probe_type] = {
            "r2_test": r2_test, "best_r2": best_r2, "best_layer": best_layer,
            "experiments": experiments, "layers": layers, "block": block,
        }
        write_csvs(HERE, probe_type, experiments, layers, r2_test, best_r2, best_layer)
        plot_best_layer_heatmap(HERE, probe_type, experiments, best_r2, best_layer)
        plot_layer_trajectories_diag(HERE, probe_type, experiments, layers, r2_test)
        plot_factor_block(HERE, probe_type, block)

    plot_comparison(HERE, all_data)
    write_summary(HERE, all_data)

    print(f"[done] outputs in {HERE}")
    for path in sorted(HERE.iterdir()):
        if path.name == "analyze.py":
            continue
        print(f"  {path.name} ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
