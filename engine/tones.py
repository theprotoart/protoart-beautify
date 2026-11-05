"""Keep the picture's own light and colour (ported from Protoart Studio's color_match.py).

The AI re-detailing often comes out a little lighter (or warmer) overall. These helpers measure the slow, broad
difference between a result and the picture it came from and take it back out, so the new fine detail stays but
the overall light and colour match the original.

Blurs run in float with torch (already loaded by the engine; no SciPy needed). Big blurs are done on a smaller
copy and scaled back up, which is exact enough for broad light changes and keeps them fast on large pictures.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter


def _gauss1d(sigma: float) -> torch.Tensor:
    r = max(1, int(3 * sigma + 0.5))
    x = torch.arange(-r, r + 1, dtype=torch.float32)
    k = torch.exp(-(x * x) / (2 * sigma * sigma))
    return k / k.sum()


def _blur_t(t: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur of a (1, C, H, W) float tensor."""
    if sigma < 0.3:
        return t
    k = _gauss1d(sigma)
    c = t.shape[1]
    r = (k.numel() - 1) // 2
    h, w = t.shape[2], t.shape[3]
    mode = "reflect" if r < h and r < w else "replicate"
    t = F.pad(t, (r, r, 0, 0), mode=mode)
    t = F.conv2d(t, k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    t = F.pad(t, (0, 0, r, r), mode=mode)
    return F.conv2d(t, k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)


def blur(a: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur of a float array (H, W) or (H, W, C); sigma in pixels."""
    two_d = a.ndim == 2
    t = torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))
    t = t[None, None] if two_d else t.permute(2, 0, 1)[None]
    h, w = t.shape[2], t.shape[3]
    step = int(sigma // 4)
    if step >= 2 and min(h, w) // step >= 8:  # big blur: on a smaller copy, then back up
        small = F.interpolate(t, size=(max(1, h // step), max(1, w // step)), mode="area")
        small = _blur_t(small, sigma / step)
        t = F.interpolate(small, size=(h, w), mode="bilinear", align_corners=False)
    else:
        t = _blur_t(t, sigma)
    out = t[0, 0] if two_d else t[0].permute(1, 2, 0)
    return out.numpy()


def keep_tones(result: Image.Image, original: Image.Image, radius: float | None = None) -> Image.Image:
    """`result`'s fine detail with `original`'s broad light and colour (original is resized to match).

    Same maths as Protoart Studio's Enhance → Keep the picture's style: result + blur(original - result).
    """
    if original.size != result.size:
        original = original.resize(result.size, Image.Resampling.LANCZOS)
    e = np.asarray(result.convert("RGB"), dtype=np.float32)
    o = np.asarray(original.convert("RGB"), dtype=np.float32)
    h, w = e.shape[:2]
    if radius is None:
        radius = max(2, min(h, w) // 160)
    out = e + blur(o - e, radius)
    return Image.fromarray(np.clip(out + 0.5, 0, 255).astype(np.uint8))


def match_to_original(result: Image.Image, original: Image.Image, mask: Image.Image,
                      radius: float | None = None) -> Image.Image:
    """Match `result`'s broad light and colour inside (and around) the white of `mask` to `original`, measured
    where the original is kept (outside the mask). Used so a redrawn face sits in the picture without a halo."""
    r = np.asarray(result.convert("RGB"), dtype=np.float32)
    o = np.asarray(original.convert("RGB").resize(result.size), dtype=np.float32)
    m_img = mask.convert("L").resize(result.size)
    h, w = r.shape[:2]
    if radius is None:
        radius = max(8, min(h, w) // 12)
    grown = np.asarray(m_img.filter(ImageFilter.MaxFilter(9)), dtype=np.float32) / 255.0
    known = (grown < 0.05).astype(np.float32)
    if known.sum() < 0.02 * h * w:  # almost everything is masked: nothing to match against
        return result
    wsum = blur(known, radius)
    corr = blur((o - r) * known[..., None], radius) / np.maximum(wsum, 1e-4)[..., None]
    return Image.fromarray(np.clip(r + corr + 0.5, 0, 255).astype(np.uint8))


def mean_luma(img: Image.Image) -> float:
    """Average brightness 0-255 (Rec. 709 weights), for the engine log."""
    a = np.asarray(img.convert("RGB").resize((256, max(1, round(256 * img.height / max(1, img.width))))),
                   dtype=np.float32)
    return float((a @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)).mean())
