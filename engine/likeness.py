"""Likeness lock: IP-Adapter Plus (SD 1.5) guiding the enhance with the photo itself.

While the AI re-details the picture, each tile is also given its own part of the photo as an image prompt
(the CLIP image features of that crop), so new detail keeps the people and things looking like they did.
Each tile gets only its own crop, never the whole picture, so large images don't grow repeated faces.

Weights (optional download, pinned like the other models):
  refiners/sd15.ip_adapter.plus        ~98 MB  (h94/IP-Adapter ip-adapter-plus_sd15, converted by refiners; Apache-2.0)
  refiners/sd21.unclip.image_encoder   ~1.26 GB (OpenCLIP ViT-H/14 image encoder; OpenRAIL++)

Memory: the image encoder is only loaded while the crops are read (a few seconds), then freed before the
diffusion passes, so the adapter adds ~100 MB to the enhance itself.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Callable

import torch
from PIL import Image


class Likeness:
    def __init__(self, upscaler: Any, ip_weights: Path, encoder_weights: Path,
                 verify: Callable[[str, Path], None]) -> None:
        from refiners.fluxion.utils import load_from_safetensors
        from refiners.foundationals.clip.image_encoder import CLIPImageEncoderH
        from refiners.foundationals.latent_diffusion.stable_diffusion_1.image_prompt import SD1IPAdapter

        verify("ip_plus", Path(ip_weights))
        verify("clip_image_h", Path(encoder_weights))
        self.up = upscaler
        self.device, self.dtype = upscaler.device, upscaler.dtype
        self.encoder_weights = Path(encoder_weights)
        weights = {k: v.to(dtype=self.dtype) for k, v in load_from_safetensors(ip_weights).items()}
        # A weightless ("meta") encoder only gives the adapter its shapes; the real one is loaded on demand.
        meta = CLIPImageEncoderH(device=torch.device("meta"), dtype=self.dtype)
        unet = upscaler.sd.unet
        # The Tile ControlNet sits inside the UNet with look-alike attention layers: unplug it while the
        # adapter finds the UNet's own cross-attention layers.
        upscaler.controlnet.eject()
        try:
            self.ip = SD1IPAdapter(target=unet, clip_image_encoder=meta, fine_grained=True, weights=weights)
        finally:
            upscaler.controlnet.inject()
        del weights
        self.ip.image_proj.to(device=self.device, dtype=self.dtype)
        self.active = False

    def enable(self, scale: float) -> None:
        self.ip.scale = float(scale)
        if not self.active:
            self.up.controlnet.eject()
            try:
                self.ip.inject()
            finally:
                self.up.controlnet.inject()
            self.active = True

    def disable(self) -> None:
        if self.active:
            self.up.controlnet.eject()
            try:
                self.ip.eject()
            finally:
                self.up.controlnet.inject()
            self.active = False

    @torch.no_grad()
    def tile_embeddings(self, crops: list[Image.Image], cfg: bool,
                        status: Callable[[str], None] | None = None) -> list[torch.Tensor]:
        """CLIP image features of each crop -> the adapter's image tokens ([negative, positive] with CFG)."""
        from refiners.foundationals.clip.image_encoder import CLIPImageEncoderH
        from refiners.foundationals.latent_diffusion.image_prompt import IPAdapter

        if status:
            status("Likeness: reading the photo")
        enc = CLIPImageEncoderH(device=self.device, dtype=self.dtype)
        enc.load_from_safetensors(self.encoder_weights)
        grid = IPAdapter.convert_to_grid_features(enc)  # the "Plus" adapter reads patch features
        try:
            first = self.ip.preprocess_image(crops[0])
            negative = self.ip.image_proj(grid(torch.zeros_like(first)))  # as in IP-Adapter Plus training
            out = []
            for crop in crops:
                positive = self.ip.image_proj(grid(self.ip.preprocess_image(crop)))
                out.append(torch.cat((negative, positive)) if cfg else positive)
            return out
        finally:
            del enc, grid
            gc.collect()
            if self.device.type == "mps":
                torch.mps.empty_cache()

    def set_embedding(self, embedding: torch.Tensor) -> None:
        self.ip.set_clip_image_embedding(embedding)
