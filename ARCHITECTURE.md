# Architecture

## Pipeline

```
                     amphion/Emilia-Dataset (streaming, multilingual)
                                       │
                                       ▼
                 sdm/data/streaming_emilia.py
                  resample → mono → 1 s chunks (B, N_chunks, T_samples)
                                       │
            ┌──────────────────────────┴──────────────────────────┐
            │                                                     │
            ▼                                                     ▼
   student_audio (XLS-R-normalized)                       teacher_audio (HF processor)
            │                                                     │
            ▼                                                     ▼
   sdm/modeling/distill_model.py                       sdm/data/teachers/<kind>.py
        DistillModel.encode()                                  Teacher.__call__()
        (XLS-R fairseq port → layer −1                         (model.forward, layer N
         → mean-pool per chunk →                                → mean-pool per chunk →
         optional projection head)                              optional LayerNorm)
            │                                                     │
            └──────────────────────┬──────────────────────────────┘
                                   ▼
                   sdm/train/run_distill.py:_masked_cos_l1
              0.5·(1 − cos(pred, target)) + 0.5·mean|pred − target|
                  masked by chunk_mask (zero out padded chunks)
                                   │
                                   ▼
                          loss.backward()
              xla_utils.optimizer_step() (xm.optimizer_step on XLA)
                  cosine LR schedule, grad clip 1.0,
                  ckpt every N steps to gs://, W&B if enabled
```

The student is **XLS-R 300M** in every production config. The teacher is whatever `cfg.teacher.kind` resolves to — that is the only thing that varies across the nine finetune configs (eight TTSDS2 teachers plus emotion2vec).

## Teacher protocol

A Teacher is any callable module that exposes:

```python
class Teacher(nn.Module):
    target_dim: int          # D, the output channel count
    pooled: str              # "chunked" today (one vector per 1 s chunk)

    @torch.no_grad()
    def __call__(
        self,
        audio: Tensor,           # (B, N_chunks, T_samples)
        chunk_mask: Tensor | None,  # (B, N_chunks), 1 for real, 0 for padding
        **ctx,                   # optional batch metadata (texts, languages, ...)
    ) -> Tensor:                 # (B, N_chunks, D)
        ...
```

Most teachers ignore ``ctx``; transcript-driven teachers like
``g2p_speaking_rate`` use ``ctx['texts']`` and ``ctx['languages']`` to
compute their targets without an audio model. The data loader populates
``ctx`` from the streaming Emilia record (see
``streaming_emilia.collate``).

The reference implementation is `sdm/data/teachers/hf_ssl.py` (used for mHuBERT-147 today and reusable for any HuggingFace SSL model with `output_hidden_states=True`). The flow is uniform:

1. Flatten `(B, N_chunks, T_samples) → (B·N_chunks, T_samples)`.
2. Run the underlying model.
3. Pull out a fixed layer's frame activations.
4. Mean-pool the frames within each chunk so the time axis collapses.
5. (Optional) Apply a parameter-free LayerNorm to the per-chunk vector — this anchors target magnitudes and noticeably stabilizes the cosine + L1 loss.
6. Mask out padded chunks via `chunk_mask`.

### Adding a Teacher

1. **Module.** Create `sdm/data/teachers/<kind>.py`. Implement the protocol above. Pin `target_dim` to the underlying model's hidden size for the chosen layer. If the teacher requires a non-16 kHz sample rate (MPM is 22.05 kHz, Whisper is 16 kHz, d-Vector and WeSpeaker want their own resamplers), do the resample inside the Teacher — `streaming_emilia.py` only emits 16 kHz mono.
2. **Dispatcher.** In `sdm/data/teachers/__init__.py`, extend `build_teacher` to map your `kind` string to the new class. The existing `hf_ssl` branch is the template.
3. **Config.** A scaffolding YAML already exists at `configs/finetune_<kind>.yaml`. Swap the placeholder values (`target_dim`, `model_id`, `layer`, `target_layernorm`) to match your implementation.
4. **Smoke.** Write a `tests/test_teachers_<kind>.py` exercising the protocol on a 2-second synthetic input. Confirm the shape, the dtype, and that `chunk_mask=0` rows come back zero.

## TTSDS2 factor taxonomy (target representations)

The TTSDS2 paper groups speech feature extractors into four **factors**. The SDM scope is to produce one finetune per non-XLSR teacher (XLS-R is the backbone, so it is not a teacher). Emotion2vec is added on top of the paper's list because it is part of the broader downstream evaluation.

| Factor | Teacher (paper) | SDM config | SDM teacher kind | Status |
|---|---|---|---|---|
| Generic | mHuBERT-147 (multilingual HuBERT) | `finetune_xlsr.yaml` | `hf_ssl` | trained |
| Generic | WavLM activations | (covered by mHuBERT for now) | — | deferred |
| Speaker | d-Vector | `finetune_dvector.yaml` | `dvector_torchscript` | trained |
| Speaker | WeSpeaker (ResNet34) | `finetune_wespeaker.yaml` | `wespeaker_resnet34` | trained |
| Prosody | WORLD F0 (mean per chunk) | `finetune_pitch.yaml` | `pyworld_f0` | trained |
| Prosody | Masked Prosody Model L7 | `finetune_mpm.yaml` | `mpm` | trained |
| Prosody | Speaking rate (G2P syllables / sec) | `finetune_speaking_rate.yaml` | `g2p_speaking_rate` | trained (replaces Allosaurus) |
| Intelligibility | wav2vec 2.0 ASR encoder | `finetune_w2v2_asr.yaml` | `hf_ctc` | trained |
| Intelligibility | Whisper-small encoder (multilingual) | `finetune_mwhisper.yaml` | `whisper_encoder` | trained |
| (Outside paper) | emotion2vec (FunASR base) | `finetune_emotion2vec.yaml` | `emotion2vec` | trained |

XLS-R itself is not in the table because it is the **backbone**; its representations are what we are reshaping via finetuning, not a target to predict.

## XLA-bf16 safety in the XLS-R port

PyTorch/XLA's bf16 path is faster than fp32 on TPU but degrades silently in a few places that show up specifically in wav2vec2-style architectures. `sdm/modeling/wav2vec2_fairseq.py` is an in-tree port that hardens against each one:

1. **Group/LayerNorm in bf16 produces NaNs at init for large vocabularies / long sequences.** The feature extractor uses `Fp32GroupNorm` and `Fp32LayerNorm`, which up-cast to fp32 internally and cast back. Norms are localized so bf16 GEMMs still dominate compute.
2. **`weight_norm` on the positional convolution generates non-finite gradients on XLA.** The HF→fairseq state-dict converter (`load_xlsr_from_hf`) bakes the parametrization out: it materializes `weight = g · v / |v|` once at load time and removes the `weight_norm` hook entirely. `_strip_pos_conv_weight_norm` is the explicit removal helper invoked on any backbone that came in with the parametrization still attached.
3. **`nn.MultiheadAttention` triggers a T/B transpose under bf16 that loses precision.** Replaced with `_SDPASelfAttention`, a thin wrapper around `F.scaled_dot_product_attention` that keeps the (B, T, C) layout end-to-end.
4. **No bf16 globally.** `XLA_USE_BF16=1` casts every fp32 op (including softmax / layernorm / RMS) to bf16, which kills training. We do not export it; bf16 compute is opted into via XLA's per-op autocast at the layers that benefit.

The same patterns apply when you add a new Teacher whose model gets _trained_ on the TPU. Today's teachers all run under `torch.no_grad()`, so they sidestep the gradient-side hazards entirely — but if a Teacher ever needs to be partially trained, the same Fp32 norm + SDPA + baked-weight-norm conventions apply.

## Loss

The training loop dispatches on `teacher.loss` in the YAML. Three modes
live in `sdm/train/run_distill.py`:

- **`cos_l1`** (default, dense embeddings): `0.5·(1 − cos(pred, target)) + 0.5·mean|pred − target|`. The cosine term is direction-only; the L1 term anchors magnitudes so the student doesn't collapse onto the unit sphere. Implemented as `_masked_cos_l1`.
- **`cos`** (direction-only embeddings: d-Vector, WeSpeaker): `1 − cos(pred, target)`. Speaker embeddings are L2-normalised by construction, so magnitude carries no information.
- **`l1`** (1-D scalar targets: pitch, speaking-rate): `mean|pred − target|`. Cosine is degenerate on a scalar.

All variants are masked by `chunk_mask`, summed across `(B, N_chunks)`, then divided by the unmasked chunk count. The teacher's `target_layernorm` knob (and, for scalar teachers, the `target_mean` / `target_scale` knobs) is the matching anchor on the target side. We picked these blends after trying straight MSE, LayerNorm-then-MSE, and pure cosine; the configured per-teacher choice was the most stable on bf16 XLA in each case.

## Checkpoints

`sdm/train/io.py` writes `final.pt` and `latest.pt` plus `step-NNNNN.pt` snapshots; paths support both local and `gs://` URIs (`gsutil` shells out — no Python google-cloud-storage dependency on TPU hosts). On XLA, `xla_utils.save_checkpoint` collects a CPU state dict on master, holds a rendezvous, then writes — no half-flushed checkpoints during preemption. `state_dict_is_finite` validates a checkpoint before we resume from it.
