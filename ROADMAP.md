# Roadmap

The repo's current state: all eight Teachers are implemented (see
[`ARCHITECTURE.md`](ARCHITECTURE.md) for the factor table). Each finetune
runs at 10 k steps on 10% of multilingual Emilia — convergence has been
mostly finished by that step count in the live mHuBERT run, so the
remaining experiments adopt the same budget.

## Phase 1 — train each finetune to a stable `final.pt`

| Order | Kind | Notes |
|---|---|---|
| 1 | `hf_ssl` (mHuBERT-147 layer 8) | Live; reference for the rest. |
| 2 | `hf_ctc` (wav2vec2 ASR encoder) | Smallest delta from `hf_ssl`. |
| 3 | `whisper_encoder` (multilingual Whisper-small) | Different processor; reads encoder hidden states directly. |
| 4 | `dvector_torchscript` | TTSDS torchscript bundle (Wav2Mel + dvector). |
| 5 | `wespeaker_resnet34` | Wraps the pyannote ResNet34 directly so it runs on TPU. |
| 6 | `pyworld_f0` | CPU-side compute in the dataloader workers; teacher returns `(B, N_chunks, 1)`. |
| 7 | `mpm` (Masked Prosody Model L7) | English-only; resamples 16 → 22.05 kHz inside the teacher. |
| 8 | `g2p_speaking_rate` | Phonemizes the dataset transcript on the fly (replaces the Allosaurus offline cache). |

**Definition of done for Phase 1:** all 8 finetunes have a stable run for
≥ 10 k steps, each producing a `final.pt` in `gs://sdm-ckpts/<experiment>/`.

## Phase 2 — cross-teacher probes

Goal: empirically validate (or refute) the TTSDS2 factor groupings by checking which teacher a given backbone's hidden states can predict via a simple probe. Expectation: representations within a factor predict each other well; cross-factor predictions are weak.

For every pair (i, j) of teachers, fit a small probe (linear ridge or 2-layer MLP) from backbone-i's hidden states (one of several layers) onto teacher-j's targets on a held-out Emilia split. The output is an 8 × 8 R² matrix per probe class and per layer. The diagonal validates each finetune itself; the off-diagonal block structure validates the factor groupings.

Scaffolding already in tree:
- `scripts/consolidate_weights.py` — pulls each experiment's `final.pt` from `gs://sdm-ckpts`, strips per-experiment heads, stages a uniform set of weights under one local directory.
- `scripts/run_linear_probes.py` — cross-prediction matrix driver. Currently waiting on the remaining Teacher implementations so it has more than one column to fill.

**Output:** an artifact (CSV or W&B report) showing the R² matrix and the chosen `(layer, probe class)` configuration that best separates factors.

## Phase 3 — wristband Gaussian regularizer

Once Phase 2 picks the layer where representations cluster cleanly per factor, add a regularizer at that layer to force its activations toward an isotropic Gaussian distribution. The motivation is downstream: TTSDS2 measures synthetic speech via the closed-form 2-Wasserstein distance between Gaussian approximations of the real and synthetic representation distributions (Section 2 of the paper). If the SDM layer used in that comparison is not approximately Gaussian, the closed-form W₂ understates true distance — adding the regularizer at training time should make the Gaussian approximation tight.

Scaffolding already in tree: `sdm/losses/wristband_gaussian.py` (and its test) carry the implementation forward from the pre-pivot phase. Wiring it into `run_distill.py` is a small change once the layer choice is locked. Plan to ablate with vs without the regularizer on the headline finetune.

**Definition of done:** TTSDS2 listening-test correlation (Spearman vs. MOS on the public listening-test set) is at least as good with the SDM checkpoints as with the original cohort of teachers — and ideally better at the layer where the regularizer is applied.
