"""4x upscalers (the pre-upscale before the AI, and INSTANT on its own) with feathered tiling (low-memory).

4x-UltraSharp is built in; the others in UPSCALERS are downloaded by the app the first time they're picked (ported
from Protoart Studio's Instant upscalers). spandrel reads each one's layout: ESRGAN, Real-ESRGAN (RRDBNet and the
compact SRVGG), RealPLKSR and DAT.
"""

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


# key -> (label, needs full precision, largest tile in input pixels or 0). DAT and RealPLKSR give wrong colours or
# black squares in half precision (Protoart Studio), and DAT's attention makes big tiles expensive.
UPSCALERS: dict[str, tuple[str, bool, int]] = {
    "esrgan": ("4x-UltraSharp", False, 0),
    "up_ultrasharp_v2": ("4x-UltraSharp V2", True, 128),
    "up_ultrasharp_v2_lite": ("4x-UltraSharp V2 Lite", True, 0),
    "up_nomos_webphoto": ("4xNomosWebPhoto", True, 0),
    "up_nomos8ksc": ("4xNomos8kSC", False, 0),
    "up_realesrgan_x4plus": ("Real-ESRGAN x4plus", False, 0),
    "up_realesr_general_x4v3": ("Real-ESRGAN general", False, 0),
    "up_animesharp": ("4x-AnimeSharp", False, 0),
    "up_realesrgan_anime6b": ("Real-ESRGAN Anime 6B", False, 0),
}


class ESRGAN:
    def __init__(self, path: Path, device: torch.device, dtype: torch.dtype, fp32: bool = False,
                 label: str = "4x-UltraSharp", max_tile: int = 0) -> None:
        from spandrel import ImageModelDescriptor, ModelLoader

        desc = ModelLoader(device="cpu").load_from_file(str(path))
        assert isinstance(desc, ImageModelDescriptor), "expected an image-to-image model"
        assert desc.scale == 4, f"expected a 4x model, got {desc.scale}x"
        self.scale = 4
        self.device = device
        self.label = label
        self.max_tile = max_tile
        # fp16 halves activation memory; fall back to fp32 if the arch can't do it (or gives wrong colours in it).
        self.dtype = dtype if (dtype == torch.float16 and desc.supports_half and not fp32) else torch.float32
        self.model = desc.model.to(device=device, dtype=self.dtype).eval()

    @torch.inference_mode()
    def _run(self, tile: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(tile).permute(2, 0, 1).unsqueeze(0).to(self.device, self.dtype)
        out = self.model(t).clamp_(0, 1)
        return out.squeeze(0).permute(1, 2, 0).float().cpu().numpy()

    def tile_for(self, tile: int) -> int:
        return min(tile, self.max_tile) if self.max_tile else tile

    def upscale(self, image: Image.Image, tile: int = 192, overlap: int = 16, on_tile=None) -> Image.Image:
        tile = self.tile_for(tile)
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

    def upscale_safe(self, image: Image.Image, tile: int = 192, overlap: int = 16, on_tile=None) -> Image.Image:
        """upscale(), retried on the CPU if the Mac's GPU can't run one of the network's operations (or runs out of
        GPU memory): slower, but it works (as in Protoart Studio)."""
        try:
            return self.upscale(image, tile=tile, overlap=overlap, on_tile=on_tile)
        except RuntimeError as e:
            if self.device.type == "cpu":
                raise
            print(f"[upscale] {self.label} on {self.device} failed ({str(e)[:200]}); retrying on the CPU", flush=True)
            gpu, dt = self.device, self.dtype
            self.to(torch.device("cpu"), torch.float32)
            try:
                return self.upscale(image, tile=tile, overlap=overlap, on_tile=on_tile)
            finally:
                self.to(gpu, dt)

    def to(self, device: torch.device, dtype: torch.dtype | None = None) -> None:
        if dtype is not None:
            self.dtype = dtype
        self.model.to(device=device, dtype=self.dtype)
        self.device = device
