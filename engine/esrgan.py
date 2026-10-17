"""4x-UltraSharp ESRGAN pre-upscaler with feathered tiling (low-memory)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


def tile_origins(h: int, w: int, tile: int, overlap: int) -> list[tuple[int, int]]:
    step = max(tile - overlap, 1)
    ys = range(0, max(h - overlap, 1), step)
    xs = range(0, max(w - overlap, 1), step)
    return [(min(y, max(h - tile, 0)), min(x, max(w - tile, 0))) for y in ys for x in xs]


class ESRGAN:
    def __init__(self, path: Path, device: torch.device, dtype: torch.dtype) -> None:
        from spandrel import ImageModelDescriptor, ModelLoader

        desc = ModelLoader(device="cpu").load_from_file(str(path))
        assert isinstance(desc, ImageModelDescriptor), "expected an image-to-image model"
        assert desc.scale == 4, f"expected a 4x model, got {desc.scale}x"
        self.scale = 4
        self.device = device
        # fp16 halves activation memory; fall back to fp32 if the arch can't do it.
        self.dtype = dtype if (dtype == torch.float16 and desc.supports_half) else torch.float32
        self.model = desc.model.to(device=device, dtype=self.dtype).eval()

    @torch.inference_mode()
    def _run(self, tile: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(tile).permute(2, 0, 1).unsqueeze(0).to(self.device, self.dtype)
        out = self.model(t).clamp_(0, 1)
        return out.squeeze(0).permute(1, 2, 0).float().cpu().numpy()

    def upscale(self, image: Image.Image, tile: int = 192, overlap: int = 16, on_tile=None) -> Image.Image:
        src = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        h, w, _ = src.shape
        s = self.scale
        acc = np.zeros((h * s, w * s, 3), dtype=np.float32)
        wsum = np.zeros((h * s, w * s, 1), dtype=np.float32)

        for y0, x0 in tile_origins(h, w, tile, overlap):
            y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
            out = self._run(src[y0:y1, x0:x1])
            th, tw = out.shape[:2]
            # Linear feather on edges that touch another tile.
            ramp = overlap * s
            wy = np.ones(th, np.float32)
            wx = np.ones(tw, np.float32)
            if ramp > 0:
                r = np.linspace(0.0, 1.0, ramp + 2, dtype=np.float32)[1:-1]
                if y0 > 0:
                    wy[:ramp] = r[: min(ramp, th)]
                if y1 < h:
                    wy[-ramp:] = np.minimum(wy[-ramp:], r[::-1][-min(ramp, th):])
                if x0 > 0:
                    wx[:ramp] = r[: min(ramp, tw)]
                if x1 < w:
                    wx[-ramp:] = np.minimum(wx[-ramp:], r[::-1][-min(ramp, tw):])
            wgt = (wy[:, None] * wx[None, :])[..., None]
            acc[y0 * s:y0 * s + th, x0 * s:x0 * s + tw] += out * wgt
            wsum[y0 * s:y0 * s + th, x0 * s:x0 * s + tw] += wgt
            if on_tile:
                on_tile()
        result = acc / np.maximum(wsum, 1e-6)
        return Image.fromarray((result * 255.0 + 0.5).clip(0, 255).astype(np.uint8))

    def to(self, device: torch.device) -> None:
        self.model.to(device)
        self.device = device
