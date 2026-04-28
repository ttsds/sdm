"""Generate analysis artefacts from a probe matrix produced by run_linear_probes.py.

Usage:
    uv run python scripts/analyze_probes.py \\
        --matrix runs/probes/20260428T150333Z/matrix.json \\
        [--out runs/probes/20260428T150333Z] \\
        [--wandb] \\
        [--wandb-run-id <id>]

When --wandb is passed the script resumes the W&B run that run_linear_probes.py
created (using the run ID stored next to matrix.json in wandb_run_id.txt if
present, or the value of --wandb-run-id) and logs images + scalar metrics into
that same run, so the table and the visuals live together.

Outputs written to --out (defaults to the directory containing matrix.json):
  summary.txt                   text report (all probe types)
  r2_full_long.csv              long-form CSV
  best_layer_matrix_<pt>.csv    R²_test at each row's best layer
  best_layer_indices_<pt>.csv   layer index that achieved that R²
  heatmap_best_layer_<pt>.png   per-row best-layer heatmap
  layer_trajectory_diag_<pt>.png  R²_test vs layer for diagonal
  factor_block_<pt>.png         factor-averaged R² heatmap
  heatmap_comparison.png        side-by-side when multiple probe types are present
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

# Short display labels for cramped plots.
SHORT = {
    "sdm-xlsr-fairseq": "xlsr",
    "sdm-xlsr":         "xlsr",
    "sdm-dvector":      "dvec",
    "sdm-pitch":        "pitch",
    "sdm-mpm":          "mpm",
    "sdm-speaking-rate":"spkrate",
    "sdm-w2v2-asr":     "w2v2-asr",
    "sdm-mwhisper":     "mwhisper",
    "sdm-emotion2vec":  "emo2vec",
}


def _ordered(experiments: set[str]) -> list[str]:
    keyed = [(FACTOR_ORDER.index(FACTOR_OF.get(e, "Other"))
              if FACTOR_OF.get(e) in FACTOR_ORDER else 999, e)
             for e in experiments]
    return [e for _, e in sorted(keyed)]


def available_probe_types(rows: list[dict]) -> list[str]:
    seen = {r.get("probe_type", "ridge") for r in rows}
    return [pt for pt in ("ridge", "mlp") if pt in seen]


def load_matrix(
    rows: list[dict], probe_type: str
) -> tuple[np.ndarray, np.ndarray, list[str], list[int]]:
    filtered = [r for r in rows if r.get("probe_type", "ridge") == probe_type]
    df = pd.DataFrame(filtered)
    experiments = _ordered(set(df["sdm"]) | set(df["target"]))
    layers = sorted(df["layer"].unique())
    n_e, n_l = len(experiments), len(layers)
    r2_test = np.full((n_e, n_e, n_l), np.nan, dtype=np.float64)
    r2_train = np.full_like(r2_test, np.nan)
    e_idx = {e: i for i, e in enumerate(experiments)}
    l_idx = {layer: i for i, layer in enumerate(layers)}
    for r in filtered:
        i, j, k = e_idx.get(r["sdm"]), e_idx.get(r["target"]), l_idx.get(r["layer"])
        if None in (i, j, k):
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


def factor_block(best_r2: np.ndarray, experiments: list[str]) -> pd.DataFrame:
    factors_present = [f for f in FACTOR_ORDER
                       if any(FACTOR_OF.get(e) == f for e in experiments)]
    df = pd.DataFrame(best_r2, index=experiments, columns=experiments).stack().reset_index()
    df.columns = ["sdm", "target", "r2"]
    df["sf"] = df["sdm"].map(FACTOR_OF)
    df["tf"] = df["target"].map(FACTOR_OF)
    return (df.groupby(["sf", "tf"])["r2"].mean().unstack("tf")
            .reindex(index=factors_present, columns=factors_present))


def _cell_labels(best_r2: np.ndarray, best_layer: np.ndarray) -> np.ndarray:
    out = np.empty(best_r2.shape, dtype=object)
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = f"{best_r2[i,j]:+.2f}\nL{int(best_layer[i,j])}"
    return out


def _draw_heatmap(ax, data: np.ndarray, row_labels, col_labels,
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
                ax.text(j, i, cell_text[i, j], ha="center", va="center",
                        fontsize=6, color="white" if abs(v) > 0.5 else "black")
    ax.set_title(title, fontsize=9)
    return im


def _short(exps: list[str]) -> list[str]:
    return [SHORT.get(e, e.replace("sdm-", "")) for e in exps]


def plot_heatmap(out: Path, probe_type: str, experiments: list[str],
                 best_r2: np.ndarray, best_layer: np.ndarray) -> Path:
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    cell = _cell_labels(best_r2, best_layer)
    im = _draw_heatmap(ax, best_r2, _short(experiments), _short(experiments), cell_text=cell,
                       title=f"Cross-teacher probe ({probe_type}) — best layer per row")
    ax.set_xlabel("teacher target"); ax.set_ylabel("backbone (SDM_i)")
    fig.colorbar(im, ax=ax, label="R² test"); fig.tight_layout()
    path = out / f"heatmap_best_layer_{probe_type}.png"
    fig.savefig(path, dpi=160); plt.close(fig)
    return path


def plot_trajectories(out: Path, probe_type: str, experiments: list[str],
                      layers: list[int], r2_test: np.ndarray) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap("tab10")
    for idx, exp in enumerate(experiments):
        i = experiments.index(exp)
        ax.plot(layers, r2_test[i, i, :], marker="o", markersize=3, lw=1.5,
                color=cmap(idx % 10), label=SHORT.get(exp, exp))
    ax.axhline(0.0, color="grey", lw=0.5, ls="--")
    ax.set_xlabel("backbone layer"); ax.set_ylabel("R² test (self)")
    ax.set_title(f"Per-layer R² — SDM_i predicting its own teacher ({probe_type})")
    ax.legend(fontsize=7, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.25))
    ax.grid(True, alpha=0.3); fig.tight_layout()
    path = out / f"layer_trajectory_diag_{probe_type}.png"
    fig.savefig(path, dpi=160, bbox_inches="tight"); plt.close(fig)
    return path


def plot_factor(out: Path, probe_type: str, block: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    cell = np.array([[f"{v:+.2f}" for v in row] for row in block.values])
    im = _draw_heatmap(ax, block.values, list(block.index), list(block.columns),
                       cell_text=cell, title=f"Factor-averaged R² ({probe_type})")
    ax.set_xlabel("target factor"); ax.set_ylabel("backbone factor")
    fig.colorbar(im, ax=ax, label="R² test (avg)"); fig.tight_layout()
    path = out / f"factor_block_{probe_type}.png"
    fig.savefig(path, dpi=160); plt.close(fig)
    return path


def plot_comparison(out: Path, all_data: dict) -> Path | None:
    probe_types = list(all_data)
    if len(probe_types) < 2:
        return None
    fig, axes = plt.subplots(1, len(probe_types), figsize=(7.5 * len(probe_types), 6.5), squeeze=False)
    for ax, pt in zip(axes[0], probe_types):
        d = all_data[pt]
        cell = _cell_labels(d["best_r2"], d["best_layer"])
        im = _draw_heatmap(ax, d["best_r2"], _short(d["experiments"]), _short(d["experiments"]),
                           cell_text=cell, title=f"{pt} — best-layer R²_test")
        ax.set_xlabel("teacher target"); ax.set_ylabel("backbone")
        fig.colorbar(im, ax=ax, label="R² test")
    fig.suptitle("Ridge vs MLP probe comparison", fontsize=11); fig.tight_layout()
    path = out / "heatmap_comparison.png"
    fig.savefig(path, dpi=160); plt.close(fig)
    return path


def write_csvs(out: Path, probe_type: str, experiments: list[str], layers: list[int],
               r2_test: np.ndarray, best_r2: np.ndarray, best_layer: np.ndarray) -> None:
    pd.DataFrame(best_r2, index=experiments, columns=experiments).to_csv(
        out / f"best_layer_matrix_{probe_type}.csv")
    pd.DataFrame(best_layer.astype(int), index=experiments, columns=experiments).to_csv(
        out / f"best_layer_indices_{probe_type}.csv")


def write_summary(out: Path, all_data: dict) -> None:
    lines = ["# Probe matrix summary\n"]
    for probe_type, d in all_data.items():
        exps, best_r2, best_layer, block = (
            d["experiments"], d["best_r2"], d["best_layer"], d["block"])
        lines.append(f"## {probe_type}\n")
        lines.append("| backbone | best layer | R²_test |")
        lines.append("|---|---:|---:|")
        for i, exp in enumerate(exps):
            lines.append(f"| {exp} | {int(best_layer[i,i])} | {best_r2[i,i]:+.3f} |")
        lines.append("")
        rows = [(best_r2[i,j], int(best_layer[i,j]), sdm, tgt)
                for i, sdm in enumerate(exps) for j, tgt in enumerate(exps) if i != j]
        lines.append("Top 5 cross-teacher:")
        for r2, layer, sdm, tgt in sorted(rows, reverse=True)[:5]:
            lines.append(f"  {sdm:>20s} → {tgt:<20s}  R²={r2:+.3f}  L{layer}")
        lines.append("\nFactor block:")
        lines.append(block.round(3).to_string())
        lines.append("\n")
    (out / "summary.txt").write_text("\n".join(lines) + "\n")


def log_to_wandb(all_data: dict, paths: dict, run_id: str | None, project: str,
                 run_name: str) -> None:
    import wandb  # noqa: PLC0415

    init_kwargs: dict = dict(project=project, job_type="probe-analysis", name=run_name)
    if run_id:
        init_kwargs.update(id=run_id, resume="allow")
    run = wandb.init(**init_kwargs)

    log: dict = {}
    for probe_type, d in all_data.items():
        exps, best_r2, best_layer, block = (
            d["experiments"], d["best_r2"], d["best_layer"], d["block"])
        # Per-backbone diagonal R² (the headline metric)
        for i, exp in enumerate(exps):
            log[f"{probe_type}/diag/{SHORT.get(exp, exp)}"] = float(best_r2[i, i])
        # Factor block scalars
        for sf in block.index:
            for tf in block.columns:
                v = block.loc[sf, tf]
                if not np.isnan(v):
                    log[f"{probe_type}/factor/{sf}_{tf}"] = float(v)
        # Images
        for label, path in paths.get(probe_type, {}).items():
            if path and path.exists():
                log[f"{probe_type}/{label}"] = wandb.Image(str(path))
    if "comparison" in paths and paths["comparison"] and paths["comparison"].exists():
        log["heatmap_comparison"] = wandb.Image(str(paths["comparison"]))

    run.log(log)
    wandb.finish()


def run(matrix_path: Path, out: Path, *, use_wandb: bool, wandb_run_id: str | None,
        wandb_project: str) -> None:
    rows = json.loads(matrix_path.read_text())
    probe_types = available_probe_types(rows)

    # Always write the long CSV with everything in it.
    pd.DataFrame(rows).to_csv(out / "r2_full_long.csv", index=False)

    all_data: dict = {}
    img_paths: dict = {}
    for probe_type in probe_types:
        r2_test, _, experiments, layers = load_matrix(rows, probe_type)
        best_r2, best_layer = best_layer_per_pair(r2_test, layers)
        blk = factor_block(best_r2, experiments)
        all_data[probe_type] = {
            "r2_test": r2_test, "best_r2": best_r2, "best_layer": best_layer,
            "experiments": experiments, "layers": layers, "block": blk,
        }
        write_csvs(out, probe_type, experiments, layers, r2_test, best_r2, best_layer)
        img_paths[probe_type] = {
            "heatmap": plot_heatmap(out, probe_type, experiments, best_r2, best_layer),
            "trajectories": plot_trajectories(out, probe_type, experiments, layers, r2_test),
            "factor_block": plot_factor(out, probe_type, blk),
        }

    img_paths["comparison"] = plot_comparison(out, all_data)
    write_summary(out, all_data)

    if use_wandb:
        run_name = f"probe-analysis-{out.name}"
        log_to_wandb(all_data, img_paths, wandb_run_id, wandb_project, run_name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matrix", type=Path, required=True,
                    help="path to matrix.json from run_linear_probes.py")
    ap.add_argument("--out", type=Path, default=None,
                    help="output directory (default: same dir as matrix.json)")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-run-id", default=None,
                    help="W&B run ID to resume (falls back to wandb_run_id.txt next to matrix)")
    ap.add_argument("--wandb-project", default="sdm")
    args = ap.parse_args()

    matrix_path = args.matrix.resolve()
    if not matrix_path.exists():
        raise SystemExit(f"matrix not found: {matrix_path}")

    out = (args.out or matrix_path.parent).resolve()
    out.mkdir(parents=True, exist_ok=True)

    # Resolve W&B run ID: explicit arg > wandb_run_id.txt beside the matrix.
    wandb_run_id = args.wandb_run_id
    if wandb_run_id is None:
        id_file = matrix_path.parent / "wandb_run_id.txt"
        if id_file.exists():
            wandb_run_id = id_file.read_text().strip() or None

    run(matrix_path, out, use_wandb=args.wandb, wandb_run_id=wandb_run_id,
        wandb_project=args.wandb_project)

    print(f"[analyze] done -> {out}")
    for p in sorted(out.iterdir()):
        if p.suffix in (".py",):
            continue
        print(f"  {p.name} ({p.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
