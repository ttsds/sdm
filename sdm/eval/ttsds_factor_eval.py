"""TTSDS adapter: wraps a finetuned sdm variant as a `Benchmark` so it can
plug into the existing `BenchmarkSuite` and use the original `wasserstein_distance`
/ `frechet_distance` utilities unchanged.

We intentionally subclass the upstream class instead of forking. The upstream
package is pulled in via the `teachers` extra (see pyproject.toml).

Construction flow:
    audio (wav, sr)  -- librosa --> 24 kHz wav
                     -- neucodec.encode_code --> FSQ ids
                     -- codes_to_input_ids --> model input
                     -- DistillationModel --> per-utterance / per-frame embedding
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from sdm.data.neucodec_dataset import codes_to_input_ids
from sdm.modeling.deberta_neucodec import SdmConfig, build_model
from sdm.modeling.distillation_heads import DistillationModel, HeadSpec


@dataclass
class SdmBenchmarkConfig:
    factor: str  # "generic" | "speaker" | "prosody" | "intelligibility"
    head_name: str  # which teacher head to read out of the multi-head model
    checkpoint_path: str | Path
    model_cfg: SdmConfig
    head_specs: list[HeadSpec]
    layer_idx: int = -1
    device: str = "cpu"
    max_length: int = 2048


class _SdmEncoder:
    """Loads the finetuned distillation model and encodes audio -> embeddings."""

    def __init__(self, cfg: SdmBenchmarkConfig) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        backbone = build_model(cfg.model_cfg)
        self.model = DistillationModel(backbone, cfg.head_specs, layer_idx=cfg.layer_idx)
        state = torch.load(cfg.checkpoint_path, map_location="cpu")
        self.model.load_state_dict(state["model"], strict=False)
        self.model.to(self.device).eval()

        from neucodec import NeuCodec  # noqa: PLC0415

        self.codec = NeuCodec.from_pretrained("neuphonic/neucodec").to(self.device).eval()

    @torch.no_grad()
    def wav_to_codes(self, wav: np.ndarray, sr: int) -> torch.Tensor:
        if sr != 24000:
            import librosa  # noqa: PLC0415

            wav = librosa.resample(wav, orig_sr=sr, target_sr=24000)
        wav_t = torch.from_numpy(wav.astype(np.float32))[None, None, :].to(self.device)
        # NeuCodec exposes the encode API as `encode_code` (mirroring `decode_code`);
        # fall back to `encode` if the version on PyPI uses that name.
        encode_fn = getattr(self.codec, "encode_code", None) or self.codec.encode
        codes = encode_fn(wav_t)
        return codes.squeeze().to(torch.long)

    @torch.no_grad()
    def embed(self, wav: np.ndarray, sr: int) -> np.ndarray:
        codes = self.wav_to_codes(wav, sr).cpu().numpy().tolist()
        input_ids, attn = codes_to_input_ids(codes, self.cfg.max_length)
        input_ids = input_ids.unsqueeze(0).to(self.device)
        attn = attn.unsqueeze(0).to(self.device)
        out = self.model(input_ids, attn)[self.cfg.head_name]
        if out.dim() == 3:  # sequence -> drop padding rows then return (T, D)
            valid = attn.squeeze(0).bool()
            return out.squeeze(0)[valid].cpu().numpy()
        return out.squeeze(0).cpu().numpy()


def make_sdm_benchmark(cfg: SdmBenchmarkConfig):
    """Build a TTSDS Benchmark subclass bound to the given sdm variant.

    Returns an instance that can be dropped into ttsds.BenchmarkSuite alongside
    the original ones. Imports the upstream `ttsds` package lazily.
    """
    from ttsds.benchmarks.benchmark import (  # noqa: PLC0415
        Benchmark,
        BenchmarkCategory,
        BenchmarkDimension,
        DeviceSupport,
    )

    cat_map = {
        "generic": BenchmarkCategory.GENERIC,
        "speaker": BenchmarkCategory.SPEAKER,
        "prosody": BenchmarkCategory.PROSODY,
        "intelligibility": BenchmarkCategory.INTELLIGIBILITY,
    }
    head_spec = next(s for s in cfg.head_specs if s.name == cfg.head_name)
    dim = BenchmarkDimension.N_DIMENSIONAL  # 1-D heads not yet supported here

    class SdmBenchmark(Benchmark):
        def __init__(self) -> None:
            super().__init__(
                name=f"Sdm/{cfg.factor}/{cfg.head_name}",
                category=cat_map[cfg.factor],
                dimension=dim,
                description=f"sdm-{cfg.factor} distilled to {cfg.head_name}",
                head=cfg.head_name,
                checkpoint=str(cfg.checkpoint_path),
                supported_devices=[DeviceSupport.CPU, DeviceSupport.GPU],
                version="0.1.0",
            )
            self._encoder = _SdmEncoder(cfg)

        def _to_device(self, device: str) -> None:
            self._encoder.device = torch.device(device)
            self._encoder.model.to(device)
            self._encoder.codec.to(device)

        def _get_distribution(self, dataset) -> np.ndarray:
            embeddings: list[np.ndarray] = []
            for wav, _ in dataset.iter_with_progress(self):
                emb = self._encoder.embed(wav, dataset.sample_rate)
                if head_spec.pooled:
                    embeddings.append(emb[None, :])
                else:
                    embeddings.append(emb)
            return np.vstack(embeddings)

    return SdmBenchmark()


__all__ = [
    "SdmBenchmarkConfig",
    "make_sdm_benchmark",
]
