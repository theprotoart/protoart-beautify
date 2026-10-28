"""Finegrain/Clarity-style enhancer: ESRGAN 4x -> resize -> tiled SD1.5 + Tile ControlNet.

Mirrors the Hugging Face Space (same checkpoints, LoRAs, negative embedding and
defaults), running on Apple Silicon (MPS) in float16 with memory profiles tuned
for 8 GB machines.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
from PIL import Image
from refiners.foundationals.latent_diffusion import solvers
from refiners.foundationals.latent_diffusion.multi_diffusion import MultiDiffusion, Size
from refiners.foundationals.latent_diffusion.stable_diffusion_1.multi_upscaler import (
    MultiUpscaler,
    UpscalerCheckpoints,
    UpscalerTarget,
)

from esrgan import ESRGAN, tile_origins
from progress import Plan, Tracker

Status = Callable[..., None]  # (message, fraction, eta_seconds=None)
ESRGAN_OVERLAP = 16
# Attention is computed in chunks of this many query tokens. At 512 px tiles (4096 tokens) this
# cuts the attention working memory ~4x, which keeps 8 GB Macs out of swap.
ATTENTION_SLICE = 1024


@dataclass(frozen=True)
class MemoryProfile:
    name: str
    max_input_short_side: int   # input is downscaled to this before enhancing
    max_output_megapixels: float
    latent_tile: tuple[int, int]  # legacy fixed tile (H, W) in latent units; unused by the planner
    tile_tokens: int              # max latent tokens per diffusion tile (planner picks the fewest tiles)
    vae_tile: int               # pixels
    esrgan_tile: int            # pixels (input side)


PROFILES = {
    # 64x64 latent tiles = 512 px, SD1.5's native size: attention stays ~0.3 GB.
    "8gb": MemoryProfile("8 GB", 512, 8.0, (64, 64), 9216, 512, 192),  # 9216 = 96x96: a 768 px square in one tile
    # Space defaults (144x112 latent tiles), for 16 GB+ machines.
    "16gb": MemoryProfile("16 GB+", 768, 16.0, (144, 112), 16384, 1024, 256),
}


def pick_device() -> tuple[torch.device, torch.dtype]:
    import platform
    # Intel Macs: PyTorch only supports Apple-silicon (or AMD) GPUs through MPS; the Intel graphics in
    # e.g. a 13" MacBook Pro may report MPS as available but is unsupported and unreliable. Use the CPU.
    if platform.machine() == "x86_64":
        return torch.device("cpu"), torch.float32
    if torch.backends.mps.is_available():
        return torch.device("mps"), torch.float16
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float16
    return torch.device("cpu"), torch.float32


def sync(device: torch.device) -> None:
    """Wait for queued GPU work so timings (and the progress bar) are honest."""
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


class AppleDPMSolver(solvers.DPMSolver):
    """DPMSolver computes its noise-schedule constants in float64, which Apple GPUs (MPS) can't
    hold. Build it on the CPU (where float64 works), cast to the working dtype, then move it.
    `rebuild()` (used per tile) goes through this constructor too."""

    def __init__(self, *args: Any, device: Any = "cpu", dtype: torch.dtype = torch.float32, **kwargs: Any) -> None:
        super().__init__(*args, device="cpu", dtype=dtype, **kwargs)
        if torch.device(device).type != "cpu":
            self.to(device=device)


class AppleLCMSolver(solvers.LCMSolver):
    """LCMSolver builds a float64 DPMSolver internally; same CPU-first workaround as above."""

    def __init__(self, *args: Any, device: Any = "cpu", dtype: torch.dtype = torch.float32, **kwargs: Any) -> None:
        super().__init__(*args, device="cpu", dtype=dtype, **kwargs)
        if torch.device(device).type != "cpu":
            self.to(device=device)


def solver_class(name: str, device: torch.device) -> type:
    if device.type == "mps" and name == "DPMSolver":
        return AppleDPMSolver
    if device.type == "mps" and name == "LCMSolver":
        return AppleLCMSolver
    return getattr(solvers, name)


class SwitchableLora:
    """A LoRA kept as its small low-rank factors and merged into / removed from the base weights on
    demand, so it costs nothing per pass and can be toggled without reloading the model.

    Matching a LoRA to the UNet's layers (refiners' auto-attach) takes ~3 minutes on an 8 GB M1, so the
    result is cached next to the LoRA file: later launches map it by layer name in about a second."""

    def __init__(self, parts: list[tuple[torch.nn.Parameter, torch.Tensor, torch.Tensor, float]]) -> None:
        self.parts = parts
        self.on = False

    @staticmethod
    def _source_tag(src: Path) -> str:
        st = src.stat()
        return f"{st.st_size}-{int(st.st_mtime)}"

    @classmethod
    def load(cls, manager: Any, name: str, src: Path, scale: float, cache: Path) -> "SwitchableLora":
        unet = manager.unet
        try:
            return cls._from_cache(unet, src, cache)
        except Exception:
            pass  # no cache yet (or stale): do the slow match once and save it
        from refiners.fluxion.utils import load_from_safetensors
        from refiners.fluxion.adapters.lora import LoraAdapter

        tensors = load_from_safetensors(src)
        expected = sum(1 for k in tensors if k.endswith(".weight")) // 2
        manager.add_loras(name, tensors=tensors, scale=scale)
        del tensors
        found: list[tuple[Any, torch.Tensor, torch.Tensor, float]] = []
        for ad in [m for m in unet.modules() if isinstance(m, LoraAdapter)]:
            lora = ad.loras.get(name)
            if lora is None:
                continue
            found.append((ad.target, lora.up.weight.detach().clone(), lora.down.weight.detach().clone(),
                          float(lora.scale)))
            ad.remove_lora(name)
            if not list(ad.lora_layers):
                ad.eject()
        if len(found) < expected * 0.95:
            raise RuntimeError(f"only {len(found)} of {expected} LoRA layers attached")
        self = cls([(t.weight, up, down, sc) for t, up, down, sc in found])
        try:
            cls._save_cache(unet, found, src, cache)
        except Exception:
            pass  # caching is only a speed-up
        return self

    @classmethod
    def _save_cache(cls, unet: Any, found: list, src: Path, cache: Path) -> None:
        import json
        from safetensors.torch import save_file

        names = {id(m): n for n, m in unet.named_modules()}
        layers, out = [], {}
        for i, (target, up, down, sc) in enumerate(found):
            layers.append([names[id(target)], sc])
            out[f"{i}.up"], out[f"{i}.down"] = up.cpu().contiguous(), down.cpu().contiguous()
        tmp = cache.with_suffix(".tmp")
        save_file(out, str(tmp), metadata={"layers": json.dumps(layers), "source": cls._source_tag(src)})
        tmp.replace(cache)

    @classmethod
    def _from_cache(cls, unet: Any, src: Path, cache: Path) -> "SwitchableLora":
        import json
        from safetensors import safe_open

        with safe_open(str(cache), framework="pt") as f:
            meta = f.metadata() or {}
            if meta.get("source") != cls._source_tag(src):
                raise ValueError("stale cache")
            modules = dict(unet.named_modules())
            parts = []
            for i, (path, sc) in enumerate(json.loads(meta["layers"])):
                w = modules[path].weight
                parts.append((w, f.get_tensor(f"{i}.up").to(w.device, w.dtype),
                              f.get_tensor(f"{i}.down").to(w.device, w.dtype), float(sc)))
        if not parts:
            raise ValueError("empty cache")
        return cls(parts)

    @torch.no_grad()
    def _apply(self, sign: float) -> None:
        for weight, up, down, s in self.parts:
            u, d = up.float(), down.float()
            if weight.ndim == 2:
                delta = u @ d
            else:
                delta = (u.flatten(1) @ d.flatten(1)).reshape(weight.shape)
            weight.add_((sign * s * delta).to(weight.dtype))

    def set(self, on: bool) -> None:
        if on != self.on:
            self._apply(1.0 if on else -1.0)
            self.on = on


def enable_attention_slicing(*models: Any, slice_size: int) -> int:
    """Turn on refiners' built-in sliced attention for every attention layer (UNet, ControlNet, VAE)."""
    from refiners.fluxion.layers.attentions import ScaledDotProductAttention

    seen: dict[int, Any] = {}
    for model in models:
        for m in model.modules():
            if isinstance(m, ScaledDotProductAttention) and id(m) not in seen:
                m.slice_size = slice_size
                seen[id(m)] = m
    return list(seen.values())


def free_memory(device: torch.device) -> None:
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


class _ReportingUpscaler(MultiUpscaler):
    """MultiUpscaler with ESRGAN pre-upscale and progress/ETA reporting."""

    esrgan: ESRGAN
    esrgan_tile: int = 192
    tracker: Tracker | None = None

    def pre_upscale(self, image: Image.Image, upscale_factor: float, **_: Any) -> Image.Image:
        tr = self.tracker

        def on_tile() -> None:
            sync(self.device)
            if tr:
                tr.esrgan_tile_done()

        image = self.esrgan.upscale(image, tile=self.esrgan_tile, overlap=ESRGAN_OVERLAP, on_tile=on_tile)
        free_memory(self.device)
        image = super().pre_upscale(image=image, upscale_factor=upscale_factor / 4)
        # Pad to a multiple of 64 px (latent multiple of 8) so the planner can use whole-image or
        # evenly sized tiles instead of many overlapping 512 px ones; the padding is cropped off after.
        self.crop_box = (0, 0, image.width, image.height)
        return pad_to_multiple(image, 64)

    low_memory: bool = False
    likeness: Any = None            # Likeness (IP-Adapter Plus) when the Likeness knob is up, else None
    likeness_status: Any = None

    def compute_targets(self, image, latent_size, tile_size, num_inference_steps, first_step,  # type: ignore[override]
                        condition_scale, clip_text_embedding):
        targets = super().compute_targets(
            image=image, latent_size=latent_size, tile_size=tile_size, num_inference_steps=num_inference_steps,
            first_step=first_step, condition_scale=condition_scale, clip_text_embedding=clip_text_embedding)
        if self.likeness is not None:
            # Each tile is prompted with its own part of the photo (never the whole picture).
            crops = [image.crop((t.tile.left * 8, t.tile.top * 8, t.tile.right * 8, t.tile.bottom * 8)) for t in targets]
            embs = self.likeness.tile_embeddings(crops, cfg=self.sd.classifier_free_guidance,
                                                 status=self.likeness_status)
            for t, e in zip(targets, embs):
                t.ip_embedding = e  # type: ignore[attr-defined]
        return targets

    def diffuse_target(self, x: torch.Tensor, step: int, target: UpscalerTarget) -> torch.Tensor:
        emb = getattr(target, "ip_embedding", None)
        if self.likeness is not None and emb is not None:
            self.likeness.set_embedding(emb)
        out = super().diffuse_target(x=x, step=step, target=target)
        sync(self.device)
        if self.low_memory and self.device.type == "mps":
            torch.mps.empty_cache()  # hand cached GPU blocks back to macOS between passes
        if self.tracker:
            self.tracker.pass_done()
        return out

    def diffuse_targets(self, noise, targets, image, first_step, autoencoder_tile_length):  # type: ignore[override]
        tr = self.tracker
        if tr:
            tr.encode_start()
        with self.sd.lda.tiled_inference(image, (autoencoder_tile_length, autoencoder_tile_length)):
            latents = self.sd.lda.tiled_image_to_latents(image)
            x = self.sd.solver.add_noise(x=latents, noise=noise, step=first_step)
            del latents
            sync(self.device)
            if tr:
                tr.encode_done()
            for step in self.sd.steps:
                x = self(x, noise=noise, step=step, targets=targets)
            if tr:
                tr.decode_start()
            result = self.sd.lda.tiled_latents_to_image(x)
            if tr:
                tr.decode_done()
            return result


def pad_to_multiple(image: Image.Image, m: int) -> Image.Image:
    import numpy as np

    w, h = image.size
    pw, ph = -(-w // m) * m, -(-h // m) * m
    if (pw, ph) == (w, h):
        return image
    arr = np.asarray(image.convert("RGB"))
    mode = "reflect" if (ph - h) < h and (pw - w) < w else "edge"
    arr = np.pad(arr, ((0, ph - h), (0, pw - w), (0, 0)), mode=mode)
    return Image.fromarray(arr)


def choose_tile(lat_h: int, lat_w: int, budget: int) -> tuple[int, int]:
    """Fewest/cheapest diffusion tiles that fit the per-tile token budget (dims multiple of 8)."""
    if lat_h * lat_w <= budget:
        return lat_h, lat_w
    best: tuple[float, int, int] | None = None
    size = Size(height=lat_h, width=lat_w)
    # Tiles stay >= 512 px (64 latent) where the image allows: SD 1.5 loses coherence below that.
    for th in range(min(64, lat_h), lat_h + 1, 8):
        for tw in range(min(64, lat_w), lat_w + 1, 8):
            tokens = th * tw
            if tokens > budget:
                continue
            n = len(MultiDiffusion.generate_latent_tiles(size=size, tile_size=Size(height=th, width=tw), min_overlap=8))
            cost = n * (tokens * (1 + tokens / 12000) + 1500)  # convs ~ tokens, attention ~ tokens^2, + per-tile overhead
            if best is None or cost < best[0]:
                best = (cost, th, tw)
    return (best[1], best[2]) if best else (64, 64)


def plan_work(width: int, height: int, factor: float, tile_tokens: int,
              esrgan_tile: int, steps: int, denoise: float) -> tuple[Plan, tuple[int, int]]:
    """Count the work the same way the pipeline will do it; returns the plan and the latent tile."""
    esrgan_tiles = len(tile_origins(height, width, esrgan_tile, ESRGAN_OVERLAP))
    # ESRGAN gives 4x; MultiUpscaler.pre_upscale then resizes by factor/4, floored to /8; then padded to /64.
    out_w = int((width * 4 * (factor / 4)) // 8 * 8)
    out_h = int((height * 4 * (factor / 4)) // 8 * 8)
    pad_w, pad_h = -(-out_w // 64) * 64, -(-out_h // 64) * 64
    lat_h, lat_w = pad_h // 8, pad_w // 8
    tile = choose_tile(lat_h, lat_w, tile_tokens)
    tiles = MultiDiffusion.generate_latent_tiles(
        size=Size(height=lat_h, width=lat_w),
        tile_size=Size(height=tile[0], width=tile[1]),
        min_overlap=8,
    )
    first_step = int(steps * (1 - denoise))
    passes = len(tiles) * max(0, steps - first_step)
    return Plan(esrgan_tiles=esrgan_tiles, passes=passes, megapixels=pad_w * pad_h / 1e6), tile


def fuse_loras(*models: Any) -> int:
    """Merge LoRA weights into the base layers once (their scales never change in this app), so each
    diffusion pass runs the plain model instead of extra low-rank layers on top of hundreds of layers."""
    from refiners.fluxion.adapters.lora import LoraAdapter

    adapters = []
    for model in models:
        for m in model.modules():
            if isinstance(m, LoraAdapter) and m not in adapters:
                adapters.append(m)
    fused = 0
    for ad in adapters:
        try:
            weight = ad.target.weight
            delta = torch.zeros(weight.shape, dtype=torch.float32, device=weight.device)
            for lora in list(ad.lora_layers):
                up = lora.up.weight.detach().float()
                down = lora.down.weight.detach().float()
                if weight.ndim == 2:
                    d = up @ down
                elif up.shape[-2:] == (1, 1):
                    d = (up.flatten(1) @ down.flatten(1)).reshape(weight.shape)
                else:
                    raise ValueError("unsupported LoRA shape")
                delta += float(lora.scale) * d
        except Exception:
            continue  # leave this one as a live adapter; results are identical either way
        with torch.no_grad():
            weight.add_(delta.to(weight.dtype))
        ad.eject()
        fused += 1
    return fused


class Enhancer:
    def __init__(self, paths: dict[str, Path], status: Status) -> None:
        self.device, self.dtype = pick_device()
        status(f"Loading models on {self.device.type.upper()} ({str(self.dtype).split('.')[-1]})", None)
        ckpt = UpscalerCheckpoints(
            unet=paths["unet"],
            clip_text_encoder=paths["clip_text_encoder"],
            lda=paths["lda"],
            controlnet_tile=paths["controlnet_tile"],
            negative_embedding=paths["negative_embedding"],
            negative_embedding_key="string_to_param.*",
            loras={
                "more_details": paths["lora_more_details"],
                "sdxl_render": paths["lora_sdxl_render"],
            },
        )
        self.up = _ReportingUpscaler(checkpoints=ckpt, device=self.device, dtype=self.dtype)
        self.up.esrgan = ESRGAN(paths["esrgan"], device=self.device, dtype=self.dtype)
        self.up.manager.update_scales({"more_details": 0.5, "sdxl_render": 1.0})  # the Space's LoRA mix
        self.fused = fuse_loras(self.up.sd.unet, self.up.sd.clip_text_encoder)
        # Turbo: distilled few-step LCM-LoRA (optional download). 8/64 = its alpha / rank.
        self.turbo: SwitchableLora | None = None
        self.turbo_error = "LCM-LoRA not downloaded"
        # Measured on an 8 GB M1 (608x768): Turbo 51 s vs Fast ~125 s, similar look; adds ~0 s to launch.
        # Set PAB_TURBO=0 to skip loading it.
        import os
        if "lcm_lora" in paths and os.environ.get("PAB_TURBO", "1") != "0":
            # The ControlNet is plugged into the UNet and has look-alike attention layers; unplug it so
            # the LoRA loader can only match the UNet's own layers, then plug it back in.
            self.up.controlnet.eject()
            try:
                src = Path(paths["lcm_lora"])
                self.turbo = SwitchableLora.load(self.up.manager, "lcm", src, 8 / 64,
                                                 src.with_name(src.stem + ".layers.safetensors"))
            except Exception as e:  # keep the app usable without Turbo
                self.turbo, self.turbo_error = None, f"{type(e).__name__}: {e}"
            finally:
                self.up.controlnet.inject()
        self.attention_layers = enable_attention_slicing(
            self.up.sd.unet, self.up.sd.lda, self.up.controlnet, slice_size=ATTENTION_SLICE
        )
        self.sliced = len(self.attention_layers)
        self.rates: dict[str, dict[str, float]] = {}  # per-profile timing memory for ETAs
        self.paths = paths
        self.likeness = None  # loaded the first time the Likeness knob is used
        free_memory(self.device)
        status("Models loaded", None)

    def _instant(self, image: Image.Image, factor: float, p: "MemoryProfile", status: Status) -> Image.Image:
        tiles = len(tile_origins(image.height, image.width, p.esrgan_tile, ESRGAN_OVERLAP))
        done = [0]
        t0 = __import__("time").monotonic()

        def on_tile() -> None:
            sync(self.device)
            done[0] += 1
            el = __import__("time").monotonic() - t0
            eta = el / done[0] * (tiles - done[0])
            status(f"Upscaling (ESRGAN 4×) — tile {done[0]}/{tiles}", min(0.99, done[0] / tiles), eta)

        status("Upscaling (ESRGAN 4×)", 0.0, None)
        try:
            big = self.up.esrgan.upscale(image, tile=p.esrgan_tile, overlap=ESRGAN_OVERLAP, on_tile=on_tile)
        finally:
            free_memory(self.device)
        w, h = round(image.width * factor), round(image.height * factor)
        return big if (w, h) == big.size else big.resize((w, h), Image.Resampling.LANCZOS)

    def enhance(
        self,
        image: Image.Image,
        status: Status,
        *,
        profile: str = "8gb",
        prompt: str = "masterpiece, best quality, highres",
        negative_prompt: str = "worst quality, low quality, normal quality",
        seed: int = 42,
        upscale_factor: float = 2.0,
        controlnet_scale: float = 0.6,
        controlnet_decay: float = 1.0,
        condition_scale: float = 6.0,
        tile_width: int | None = None,
        tile_height: int | None = None,
        denoise_strength: float = 0.35,
        num_inference_steps: int = 18,
        solver: str = "DDIM",
        max_output_side: int | None = None,
        turbo: bool = False,
        # Measured on an 8 GB M1: identical output, 62 s vs 121 s, because freeing the GPU cache
        # between passes keeps macOS out of swap. On by default.
        low_memory: bool = True,
        mode: str = "detail",  # "detail" (diffusion) or "instant" (ESRGAN only, no new detail)
        likeness: float = 0.0,  # 0 = off; >0 = IP-Adapter Plus strength (keeps the photo's people and things)
    ) -> Image.Image:
        p = PROFILES.get(profile, PROFILES["8gb"])
        self.up.esrgan_tile = p.esrgan_tile

        # 0. Cap the input size (the Space uses 768; 512 keeps 8 GB Macs comfortable).
        short = min(image.size)
        if short > p.max_input_short_side:
            s = p.max_input_short_side / short
            image = image.resize((round(image.width * s), round(image.height * s)), Image.Resampling.LANCZOS)


        # 1. Optional cap on the output's long side (batch mode default: 768 px).
        #    Big inputs are shrunk so a <=2x detail pass lands exactly on the cap.
        if max_output_side:
            long = max(image.size)
            if long * upscale_factor > max_output_side:
                if long <= max_output_side:  # just lower the factor
                    upscale_factor = max_output_side / long
                else:  # shrink the input so a <=2x pass lands on the cap
                    upscale_factor = max(1.0, min(upscale_factor, 2.0))
                    s = max_output_side / upscale_factor / long
                    image = image.resize((max(8, round(image.width * s)), max(8, round(image.height * s))),
                                         Image.Resampling.LANCZOS)

        # 2. Cap the output size so a big factor can't blow up time/memory.
        max_px = p.max_output_megapixels * 1_000_000
        if image.width * image.height * upscale_factor ** 2 > max_px:
            upscale_factor = (max_px / (image.width * image.height)) ** 0.5
            status(f"Upscale factor limited to {upscale_factor:.2f}× by the {p.name} profile", None)
        upscale_factor = max(1.0, upscale_factor)

        # Instant: the 4x-UltraSharp upscaler alone, then resize to the target. Seconds, not minutes, but it
        # only sharpens (no invented detail). Good for previews and big batches.
        if mode == "instant":
            return self._instant(image, upscale_factor, p, status)

        # Low-memory: smaller attention chunks and give GPU cache back to macOS after every pass.
        self.up.low_memory = bool(low_memory)
        for layer in self.attention_layers:
            layer.slice_size = ATTENTION_SLICE // 2 if low_memory else ATTENTION_SLICE

        # Turbo = distilled LCM-LoRA: few steps and no negative-prompt pass (half the work per step).
        use_turbo = bool(turbo and self.turbo is not None)
        if turbo and not use_turbo:
            status(f"Turbo unavailable ({self.turbo_error}); using the normal engine", None)
        if self.turbo is not None:
            self.turbo.set(use_turbo)
        self.up.sd.classifier_free_guidance = not use_turbo
        if use_turbo:
            solver = "LCMSolver"
            num_inference_steps = min(int(num_inference_steps), 8)

        # Likeness lock (optional): IP-Adapter Plus guided by each tile's own crop of the photo.
        use_likeness = float(likeness or 0) > 0
        if use_likeness:
            if self.likeness is None:
                ip_w, enc_w = self._likeness_files()
                if ip_w is None:
                    raise RuntimeError("The Likeness model isn't downloaded yet. Quit and reopen the app to "
                                       "download it, or turn Likeness off.")
                status("Loading the Likeness model (first use)", None)
                from likeness import Likeness
                from models import _verify
                self.likeness = Likeness(self.up, ip_w, enc_w, _verify)
            self.likeness.enable(min(1.0, float(likeness)))
            self.up.likeness = self.likeness
            self.up.likeness_status = lambda m: status(m, None)
        elif self.likeness is not None:
            self.likeness.disable()
            self.up.likeness = None

        steps = int(num_inference_steps)
        plan, (th, tw) = plan_work(image.width, image.height, upscale_factor, p.tile_tokens, p.esrgan_tile,
                                   steps, denoise_strength)
        rate_key = profile + ("-turbo" if use_turbo else "")
        tracker = Tracker(plan=plan, emit=status, memory=self.rates.setdefault(rate_key, {}))
        self.up.tracker = tracker
        tracker._send("Pre-upscaling (ESRGAN 4×)")

        gen_device = self.device if self.device.type in ("mps", "cuda") else torch.device("cpu")
        generator = torch.Generator(device=gen_device)
        generator.manual_seed(int(seed))

        try:
            result = self.up.upscale(
                image=image,
                prompt=prompt or "",
                negative_prompt=negative_prompt or "",
                upscale_factor=upscale_factor,
                controlnet_scale=controlnet_scale,
                controlnet_scale_decay=controlnet_decay,
                condition_scale=condition_scale,
                tile_size=(th, tw),
                denoise_strength=denoise_strength,
                num_inference_steps=steps,
                loras_scale=None,  # already merged into the weights at load
                solver_type=solver_class(solver, self.device),
                autoencoder_tile_length=p.vae_tile,
                generator=generator,
            )
            tracker.finish()
            box = getattr(self.up, "crop_box", None)
            if box and (box[2], box[3]) != result.size:
                result = result.crop(box)
        finally:
            self.up.tracker = None
            if self.likeness is not None:
                self.likeness.disable()
                self.up.likeness = None
            free_memory(self.device)
        return result

    def _likeness_files(self) -> tuple[Path | None, Path | None]:
        from models import DOWNLOADS
        ip_w = self.paths.get("ip_plus") or (DOWNLOADS / "ip_plus.safetensors")
        enc_w = self.paths.get("clip_image_h") or (DOWNLOADS / "clip_image_h.safetensors")
        if not Path(ip_w).exists() or not Path(enc_w).exists():
            return None, None
        return Path(ip_w), Path(enc_w)
