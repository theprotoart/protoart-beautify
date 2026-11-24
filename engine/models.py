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
    url: str = ""           # set for files from a GitHub release: the exact address (repo/revision describe it)


REMOTES: list[Remote] = [
    Remote("unet", "refiners/sd15.juggernaut.reborn.unet", "model.safetensors",
           "347d14c3c782c4959cc4d1bb1e336d19f7dda4d2", True, 3440),
    Remote("clip_text_encoder", "refiners/sd15.juggernaut.reborn.text_encoder", "model.safetensors",
           "744ad6a5c0437ec02ad826df9f6ede102bb27481", True, 490),
    Remote("lda", "refiners/sd15.juggernaut.reborn.autoencoder", "model.safetensors",
           "3c1aae3fc3e03e4a2b7e0fa42b62ebb64f1a4c19", True, 330),
    Remote("controlnet_tile", "refiners/sd15.controlnet.tile", "model.safetensors",
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
    # Likeness lock (optional): IP-Adapter Plus SD 1.5 and the OpenCLIP ViT-H image encoder, both as converted
    # and published by refiners (SHA-256 match refiners' own conversion records).
    Remote("ip_plus", "refiners/sd15.ip_adapter.plus", "model.safetensors",
           "2516bc6c2cbd933e7949f7c2c0f076790ff3bf35", False, 94, optional=True),
    Remote("clip_image_h", "refiners/sd21.unclip.image_encoder", "model.safetensors",
           "4a36eec8c6fcfa9911a9b3290c707b8ad7591cc3", False, 1206, optional=True),
    # Upscalers you can pick instead of 4x-UltraSharp (as in Protoart Studio's Instant), each downloaded the first
    # time it's used from its author's own release. All 4x; spandrel reads every layout (DAT, RealPLKSR, ESRGAN,
    # Real-ESRGAN), with .pth files through its restricted unpickler, and every file is hash-checked before each load.
    Remote("up_ultrasharp_v2", "Kim2091/UltraSharpV2", "4x-UltraSharpV2.safetensors",
           "2d1db39ff692c111da92112d49f833e4ad7035ae", False, 134, optional=True),
    Remote("up_ultrasharp_v2_lite", "Kim2091/UltraSharpV2", "4x-UltraSharpV2_Lite.safetensors",
           "2d1db39ff692c111da92112d49f833e4ad7035ae", False, 29, optional=True),
    Remote("up_nomos_webphoto", "github:Phhofm/models", "4xNomosWebPhoto_RealPLKSR.safetensors",
           "4xNomosWebPhoto_RealPLKSR", False, 29, optional=True,
           url="https://github.com/Phhofm/models/releases/download/4xNomosWebPhoto_RealPLKSR/"
               "4xNomosWebPhoto_RealPLKSR.safetensors"),
    Remote("up_nomos8ksc", "github:Phhofm/models", "4xNomos8kSC.pth", "4xNomos8kSC", False, 64, optional=True,
           url="https://github.com/Phhofm/models/releases/download/4xNomos8kSC/4xNomos8kSC.pth"),
    Remote("up_realesrgan_x4plus", "github:xinntao/Real-ESRGAN", "RealESRGAN_x4plus.pth", "v0.1.0", False, 64,
           optional=True, url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"),
    Remote("up_realesr_general_x4v3", "github:xinntao/Real-ESRGAN", "realesr-general-x4v3.pth", "v0.2.5.0", False, 5,
           optional=True,
           url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"),
    Remote("up_animesharp", "Kim2091/AnimeSharp", "4x-AnimeSharp.safetensors",
           "7696d95ced82b0c1f2a41f6ac73336133f0a90e1", False, 32, optional=True),
    Remote("up_realesrgan_anime6b", "github:xinntao/Real-ESRGAN", "RealESRGAN_x4plus_anime_6B.pth", "v0.2.2.4", False,
           18, optional=True,
           url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"),
    # Remove spots (LaMa, advimman/lama, Apache-2.0): the big-lama generator as safetensors (data only, never code),
    # read by spandrel. Downloaded the first time Remove is used.
    Remote("lama", "michaelgold/big-lama", "big-lama.safetensors",
           "98ab7cface89a594b381f32fc7a9f2cae3f91ccc", False, 195, optional=True),
]


# SHA-256 of every model file (they match Hugging Face's own LFS hashes at the pinned commits). Each file is
# checked before it is used: the big safetensors once, before their one-time fp16 conversion (only the fp16 copy
# is kept), and the pickle-format files (.pth/.pt) before every load, because a tampered pickle can run code on
# older PyTorch. A file that doesn't match is deleted and never used.
PINNED_SHA256 = {
    "unet": "64a75d16fdb11faeedbef8270fcdfe3051284f743cdf46d8bb89c09499a22591",
    "clip_text_encoder": "c649e079cbef5ccd79ef643acac29363d153d5a5f719e9c2e1893c96ec8b2357",
    "lda": "6f1ed875201344031f2a9ddee3ff40f455eb2a5ee4833070061a5d163cb23595",
    "controlnet_tile": "3002029df75364fcbdbf6024dc4c414c929cb8623a2fe7d406a4dbfcee1ffa5a",
    "esrgan": "a5812231fc936b42af08a5edba784195495d303d5b3248c24489ef0c4021fe01",
    "negative_embedding": "979ea73200c7c29ba884e642e544c54abfaad4c9a82582c68f3212d390d41397",
    "lora_more_details": "d9cf2f88dec53a8dd8c09aba2b1d67963b265ebdf8255660f5bcd7f1c8edba33",
    "lora_sdxl_render": "672fcccf2ce6ca4ef71762a3ec1b8c8afd79328067e13c8f9756260ec94d1608",
    "lcm_lora": "8f90d840e075ff588a58e22c6586e2ae9a6f7922996ee6649a7f01072333afe4",
    "restoreformer": "07404d446d62ca3d5ed38b1de09a947a1e77d46dbccec961a74d713a8f24ace0",
    "codeformer": "1009e537e0c2a07d4cabce6355f53cb66767cd4b4297ec7a4a64ca4b8a5684b7",
    "ip_plus": "6eae5d2098fa83e3b8bf416fb46dd77a921ad044650f13890e8d41d7c84a71d2",
    "clip_image_h": "c9f43e359e06f1a237324c4c11734d6acd7fbddbfd3b1ed4f1b525267bedb812",
    "up_ultrasharp_v2": "2b5db674aa8f3864a696dccd1d07d7cc0500fbf220f35044cd30eee9a7f971a7",
    "up_ultrasharp_v2_lite": "7a330ef6b7c632477d311ac03499c91c350e35c5986d4a6c295f0628cb871a09",
    "up_nomos_webphoto": "9be0228f98156a100d6636d99b373ed2785b999723f9adc4cca504329ab157f2",
    "up_nomos8ksc": "d73b89267320411c7eb0ec7ab56ab17f1044f2408fed211588cee845bee48407",
    "up_realesrgan_x4plus": "4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1",
    "up_realesr_general_x4v3": "8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292",
    "up_animesharp": "7fc60054d291579acac4fb537efd91c80611cbd281ef8b90f434048cc1313b39",
    "up_realesrgan_anime6b": "f872d837d3c90ed2e05227bed711af5671a6fd1c9f7d7e91c911a61f155e99da",
    "lama": "326ae1889b92d07c8819c35eaf561f8b290d057536eeb12524b72997f31005ec",
}


def downloaded(key: str) -> Path | None:
    """An optional file the app has downloaded (checked against its pinned SHA-256 by the caller)."""
    r = next((x for x in REMOTES if x.key == key), None)
    if r is None:
        return None
    p = DOWNLOADS / f"{r.key}{Path(r.filename).suffix}"
    return p if p.exists() else None


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
            status(f"Checking {r.key} (one time) — {i + 1}/{total}", i / total)
            _verify(r.key, local)  # before conversion: a file that fails is deleted, never converted or used
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
