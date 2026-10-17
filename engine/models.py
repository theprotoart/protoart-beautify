"""Model download + one-time fp16 conversion.

Same checkpoints (and pinned revisions) as the finegrain-image-enhancer Space.
Weights are converted to float16 once after download so that loading them on an
8 GB Mac never needs a full float32 copy in memory.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

APP_SUPPORT = Path(os.environ.get(
    "PAB_DATA_DIR",
    Path.home() / "Library" / "Application Support" / "ProtoArt Beautify",
))
HF_CACHE = APP_SUPPORT / "hf-cache"
DOWNLOADS = APP_SUPPORT / "downloads"  # filled by the Mac app's native downloader
FP16_DIR = APP_SUPPORT / "models-fp16"

os.environ.setdefault("HF_HOME", str(HF_CACHE))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


@dataclass(frozen=True)
class Remote:
    key: str
    repo_id: str
    filename: str
    revision: str
    convert_fp16: bool  # safetensors we can shrink to fp16 on disk
    approx_mb: int
    optional: bool = False  # skipped (not downloaded here) if the app hasn't fetched it


REMOTES: list[Remote] = [
    Remote("unet", "refiners/juggernaut.reborn.sd1_5.unet", "model.safetensors",
           "347d14c3c782c4959cc4d1bb1e336d19f7dda4d2", True, 3440),
    Remote("clip_text_encoder", "refiners/juggernaut.reborn.sd1_5.text_encoder", "model.safetensors",
           "744ad6a5c0437ec02ad826df9f6ede102bb27481", True, 490),
    Remote("lda", "refiners/juggernaut.reborn.sd1_5.autoencoder", "model.safetensors",
           "3c1aae3fc3e03e4a2b7e0fa42b62ebb64f1a4c19", True, 330),
    Remote("controlnet_tile", "refiners/controlnet.sd1_5.tile", "model.safetensors",
           "48ced6ff8bfa873a8976fa467c3629a240643387", True, 1450),
    Remote("esrgan", "philz1337x/upscaler", "4x-UltraSharp.pth",
           "011deacac8270114eb7d2eeff4fe6fa9a837be70", False, 67),
    Remote("negative_embedding", "philz1337x/embeddings", "JuggernautNegative-neg.pt",
           "203caa7e9cc2bc225031a4021f6ab1ded283454a", False, 1),
    Remote("lora_more_details", "philz1337x/loras", "more_details.safetensors",
           "a3802c0280c0d00c2ab18d37454a8744c44e474e", False, 10),
    Remote("lora_sdxl_render", "philz1337x/loras", "SDXLrender_v2.0.safetensors",
           "a3802c0280c0d00c2ab18d37454a8744c44e474e", False, 10),
    # Distilled few-step add-on (Latent Consistency LoRA) used by Turbo mode.
    Remote("lcm_lora", "latent-consistency/lcm-lora-sdv1-5", "pytorch_lora_weights.safetensors",
           "cf2fced511dbe7e26c8d1d397e728fbab875db4b", False, 135, optional=True),
    # Face fix (optional): RestoreFormer (TencentARC GFPGAN v1.3.4 release, Apache-2.0) and CodeFormer
    # (sczhou/CodeFormer v0.1.0 release, S-Lab License 1.0, non-commercial), from a Hugging Face mirror at a
    # pinned commit. Both are byte-identical to the official GitHub release files (same SHA-256).
    Remote("restoreformer", "fofr/comfyui", "facerestore_models/RestoreFormer.pth",
           "173115bf2afcefcc7d6b27aedc681a5f5c9c8b0b", False, 278, optional=True),
    Remote("codeformer", "fofr/comfyui", "facerestore_models/codeformer.pth",
           "173115bf2afcefcc7d6b27aedc681a5f5c9c8b0b", False, 360, optional=True),
]


# SHA-256 of files used exactly as downloaded (match Hugging Face's LFS hashes). The two pickle-format
# files (.pth/.pt) are always checked before loading: a tampered pickle can run code on older PyTorch.
PINNED_SHA256 = {
    "esrgan": "a5812231fc936b42af08a5edba784195495d303d5b3248c24489ef0c4021fe01",
    "negative_embedding": "979ea73200c7c29ba884e642e544c54abfaad4c9a82582c68f3212d390d41397",
    "lora_more_details": "d9cf2f88dec53a8dd8c09aba2b1d67963b265ebdf8255660f5bcd7f1c8edba33",
    "lora_sdxl_render": "672fcccf2ce6ca4ef71762a3ec1b8c8afd79328067e13c8f9756260ec94d1608",
    "lcm_lora": "8f90d840e075ff588a58e22c6586e2ae9a6f7922996ee6649a7f01072333afe4",
    "restoreformer": "07404d446d62ca3d5ed38b1de09a947a1e77d46dbccec961a74d713a8f24ace0",
    "codeformer": "1009e537e0c2a07d4cabce6355f53cb66767cd4b4297ec7a4a64ca4b8a5684b7",
}


def _sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify(key: str, path: Path) -> None:
    want = PINNED_SHA256.get(key)
    if want and _sha256(path) != want:
        try:
            path.unlink()  # the app downloads a fresh copy on the next launch
        except OSError:
            pass
        raise RuntimeError(f"The model file '{key}' failed its security check and was removed. "
                           "Quit and reopen the app to download it again.")


def _to_fp16(src: Path, dst: Path) -> None:
    """Stream tensors one at a time so peak RAM stays ~ the fp16 size."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    out: dict[str, "torch.Tensor"] = {}
    with safe_open(str(src), framework="pt", device="cpu") as f:
        meta = f.metadata()
        for name in f.keys():
            t = f.get_tensor(name)
            if t.is_floating_point() and t.dtype != torch.float16:
                t = t.to(torch.float16)
            out[name] = t.contiguous()
    tmp = dst.with_suffix(".tmp")
    save_file(out, str(tmp), metadata=meta)
    tmp.replace(dst)


def ensure_models(status: Callable[[str, float | None], None]) -> dict[str, Path]:
    """Return local paths for every checkpoint, converting big ones to fp16 the first time.

    The Mac app downloads the files into DOWNLOADS itself; the Hugging Face download here is
    only a fallback for the command-line tool.
    """
    FP16_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    total = len(REMOTES)
    for i, r in enumerate(REMOTES):
        final = FP16_DIR / f"{r.key}.fp16.safetensors"
        if r.convert_fp16 and final.exists():
            paths[r.key] = final
            continue
        raw = DOWNLOADS / f"{r.key}{Path(r.filename).suffix}"
        from_app = raw.exists()
        if r.optional and not from_app:
            continue
        if from_app:
            local = raw
        else:
            from huggingface_hub import hf_hub_download
            status(f"Downloading {r.key} (~{r.approx_mb} MB) — {i + 1}/{total}", i / total)
            local = Path(hf_hub_download(repo_id=r.repo_id, filename=r.filename, revision=r.revision))
        if r.convert_fp16:
            status(f"Converting {r.key} to fp16 (one time) — {i + 1}/{total}", i / total)
            _to_fp16(local, final)
            # Drop the big fp32 original: we only ever load the fp16 copy.
            try:
                if from_app:
                    raw.unlink()
                else:
                    real = local.resolve()
                    real.unlink()
                    local.unlink(missing_ok=True)
            except OSError:
                pass
            paths[r.key] = final
        else:
            _verify(r.key, local)
            paths[r.key] = local
    status("Models ready", 1.0)
    return paths
