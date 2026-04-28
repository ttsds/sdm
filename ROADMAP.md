# Roadmap

The repo's current state: **Phase 1 is done.** All nine teachers (the eight
TTSDS2 teachers plus emotion2vec) have been trained for 10 k steps on 10%
of multilingual Emilia. Each `final.pt` lives at
`gs://sdm-ckpts/<experiment>/final.pt`. Convergence was mostly finished by
that step count in the reference mHuBERT run, so the remaining experiments
inherited the same budget.

The current focus is **Phase 2 — cross-teacher probes.**

## Phase 1 — train each finetune to a stable `final.pt` *(done)*

| # | Kind | Config | Notes |
|---|---|---|---|
| 1 | `hf_ssl` (mHuBERT-147 layer 8) | `finetune_xlsr.yaml` | reference; first run, others adopted its budget. |
| 2 | `hf_ctc` (wav2vec2 ASR encoder) | `finetune_w2v2_asr.yaml` | smallest delta from `hf_ssl`. |
| 3 | `whisper_encoder` (multilingual Whisper-small) | `finetune_mwhisper.yaml` | different processor; reads encoder hidden states directly. |
| 4 | `dvector_torchscript` | `finetune_dvector.yaml` | TTSDS torchscript bundle (Wav2Mel + dvector); cosine loss. |
| 5 | `wespeaker_resnet34` | `finetune_wespeaker.yaml` | wespeaker-unofficial multilingual checkpoint; cosine loss. |
| 6 | `pyworld_f0` | `finetune_pitch.yaml` | CPU-side compute in dataloader workers; 1-D scalar target. |
| 7 | `mpm` (Masked Prosody Model L7) | `finetune_mpm.yaml` | English-only; resamples 16 → 22.05 kHz inside the teacher. |
| 8 | `g2p_speaking_rate` | `finetune_speaking_rate.yaml` | phonemizes the dataset transcript (replaces the Allosaurus offline cache). |
| 9 | `emotion2vec` | `finetune_emotion2vec.yaml` | FunASR base model; outside the paper but part of downstream eval. |

**Done criterion (met):** every finetune has a stable run for ≥ 10 k steps
and a `final.pt` in `gs://sdm-ckpts/<experiment>/`.

Provisioning lived in `.tpu/provision_<short>.sh` (now committed; secrets
sourced from `.env`). Fan-out via `.tpu/provision_all.sh`; teardown via
`.tpu/teardown.sh`.

## Phase 2 — cross-teacher probes *(active)*

Goal: empirically validate (or refute) the TTSDS2 factor groupings by
checking which teacher a given backbone's hidden states can predict via a
simple probe. Expectation: representations within a factor predict each
other well; cross-factor predictions are weak.

For every pair (i, j) of the nine experiments, fit a small probe (linear
ridge or 2-layer MLP) from backbone-i's hidden states (one of several
layers) onto teacher-j's chunk targets on a held-out Emilia split. The
output is a 9 × 9 R² matrix per probe class and per layer. The diagonal
validates each finetune itself; the off-diagonal block structure validates
the factor groupings.

Pieces in tree:

- `scripts/consolidate_weights.py` — pulls each experiment's `final.pt`
  from `gs://sdm-ckpts`, strips per-experiment heads, stages a uniform set
  of weights under one local directory.
- `scripts/run_linear_probes.py` — cross-prediction matrix driver
  (Ridge / R² across the 9 × 9 grid; `--layer-sweep` for per-layer probes).
  Default held-out: 100 utterances from `mythicinfinity/libritts dev.clean`.
  Small probe set on purpose — this is a sanity check on the factor
  groupings, not a precision benchmark; we'll scale up once the matrix
  shape stabilises and pick a multilingual split.
- `scripts/run_probes.sh` — single-TPU driver: consolidate the 9
  checkpoints, compute teacher targets + backbone features once each on
  the held-out batch, fit probes, upload `matrix.json` to
  `gs://sdm-ckpts/probes/<run-id>/`.
- `.tpu/provision_probes.sh` — provisions a single `sdm-probes` v6e-8 TPU
  and launches `run_probes.sh` in tmux. Backbone + teacher forwards run on
  TPU; Ridge fits on the host CPU.

**Output:** an artifact (CSV or W&B report) showing the R² matrix and the
chosen `(layer, probe class)` configuration that best separates factors.
That choice locks the layer used for Phase 3.

## Phase 3 — wristband Gaussian regularizer

Once Phase 2 picks the layer where representations cluster cleanly per
factor, add a regularizer at that layer to force its activations toward an
isotropic Gaussian distribution. The motivation is downstream: TTSDS2
measures synthetic speech via the closed-form 2-Wasserstein distance
between Gaussian approximations of the real and synthetic representation
distributions (Section 2 of the paper). If the SDM layer used in that
comparison is not approximately Gaussian, the closed-form W₂ understates
true distance — adding the regularizer at training time should make the
Gaussian approximation tight.

Scaffolding already in tree: `sdm/losses/wristband_gaussian.py` (and its
test) carry the implementation forward from the pre-pivot phase. Wiring it
into `run_distill.py` is a small change once the layer choice is locked.
Plan to ablate with vs without the regularizer on the headline finetune.

**Done criterion:** TTSDS2 listening-test correlation (Spearman vs. MOS on
the public listening-test set) is at least as good with the SDM
checkpoints as with the original cohort of teachers — and ideally better
at the layer where the regularizer is applied.
