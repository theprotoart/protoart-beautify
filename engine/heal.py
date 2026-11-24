"""Remove spots (ported from Protoart Studio's 🩹 Remove): paint over a scratch, dust, a blemish or something in the
way, and it's filled in from its surroundings, like Photoshop's healing brush.

It uses LaMa (advimman/lama, Apache-2.0), a model made for exactly this: it sees the whole area around the painted
part, so it continues textures, lines and backgrounds. ~195 MB, downloaded by the app the first time Remove is used,
as safetensors (data only) and checked against its pinned SHA-256 before it's loaded. spandrel (already in the engine)
runs it, on the Apple GPU when it can and otherwise on the CPU: a few seconds, no other model needs to be loaded.

Only the painted area changes: the result is blended back into the untouched picture with a soft edge.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

WORK_MAX = 1024     # the area LaMa looks at is worked at most this big (memory; it was trained at 256-512 px)
MIN_CONTEXT = 256   # and at least this big, so a tiny spot still sees its surroundings

_model = None       # (descriptor, device)


def load(path, device):
    """LaMa from its safetensors file (the caller checks the hash first)."""
    global _model
    if _model is None:
        import torch
        from spandrel import MaskedImageModelDescriptor, ModelLoader
        desc = ModelLoader(device="cpu").load_from_file(str(path))
        if not isinstance(desc, MaskedImageModelDescriptor):
            raise RuntimeError("The Remove model isn't a LaMa inpainting model.")
        desc.model.float().eval()
        dev = device if getattr(device, "type", "cpu") in ("mps", "cuda") else torch.device("cpu")
        desc.model.to(dev)
        _model = (desc, dev)
    return _model


def box(mask: np.ndarray, pad_frac: float = 0.6, min_side: int = MIN_CONTEXT) -> tuple[int, int, int, int]:
    """The part of the picture LaMa looks at: the painted area plus plenty of surroundings (it needs context),
    square-ish, inside the picture. -> (x0, y0, x1, y1)"""
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    side = max(x1 - x0, y1 - y0)
    side = int(max(min_side, side * (1 + 2 * pad_frac)))
    side = min(side, max(w, h))
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    bx0 = int(np.clip(cx - side // 2, 0, max(0, w - side)))
    by0 = int(np.clip(cy - side // 2, 0, max(0, h - side)))
    return bx0, by0, min(w, bx0 + side), min(h, by0 + side)


def _run(desc, device, crop: Image.Image, crop_mask: Image.Image) -> Image.Image:
    """LaMa on one crop, worked at <= WORK_MAX px (multiples of 8), back at the crop's size."""
    import torch
    w, h = crop.size
    s = min(1.0, WORK_MAX / max(w, h))
    ww, hh = max(8, round(w * s / 8) * 8), max(8, round(h * s / 8) * 8)
    img = np.asarray(crop.resize((ww, hh), Image.Resampling.LANCZOS), dtype=np.float32) / 255.0
    m = (np.asarray(crop_mask.resize((ww, hh), Image.Resampling.NEAREST), dtype=np.float32) > 127).astype(np.float32)

    def go(dev):
        x = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(dev)
        mk = torch.from_numpy(m).unsqueeze(0).unsqueeze(0).to(dev)
        with torch.inference_mode():
            return desc(x, mk).squeeze(0).clamp(0, 1).float().cpu().numpy().transpose(1, 2, 0)

    try:
        out = go(device)
    except RuntimeError as e:  # an operation the Mac's GPU can't do (the FFT layers): the CPU works
        if device.type == "cpu":
            raise
        print(f"[heal] LaMa on {device} failed ({str(e)[:160]}); using the CPU", flush=True)
        desc.model.to("cpu")
        global _model
        _model = (desc, torch.device("cpu"))
        out = go(torch.device("cpu"))
    res = Image.fromarray((out * 255.0 + 0.5).clip(0, 255).astype(np.uint8))
    return res.resize((w, h), Image.Resampling.LANCZOS) if (ww, hh) != (w, h) else res


def regions(mask: np.ndarray, grid: int = 192) -> list[np.ndarray]:
    """The painted area split into separate spots (connected on a coarse grid), each as a full-size bool mask, so
    spots far apart are each healed close up instead of the whole picture being shrunk to fit."""
    h, w = mask.shape
    f = max(1, int(np.ceil(max(h, w) / grid)))
    gh, gw = -(-h // f), -(-w // f)
    pad = np.zeros((gh * f, gw * f), bool)
    pad[:h, :w] = mask
    coarse = pad.reshape(gh, f, gw, f).any(axis=(1, 3))
    labels = np.zeros(coarse.shape, np.int32)
    n = 0
    for sy, sx in zip(*np.nonzero(coarse)):
        if labels[sy, sx]:
            continue
        n += 1
        labels[sy, sx] = n
        stack = [(sy, sx)]
        while stack:
            y, x = stack.pop()
            for dy in (-1, 0, 1):          # 8-connected, and spots one cell apart count as one
                for dx in (-1, 0, 1):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < gh and 0 <= xx < gw and coarse[yy, xx] and not labels[yy, xx]:
                        labels[yy, xx] = n
                        stack.append((yy, xx))
    out = []
    for k in range(1, n + 1):
        full = np.repeat(np.repeat(labels == k, f, axis=0), f, axis=1)[:h, :w]
        part = mask & full
        if part.any():
            out.append(part)
    return out


def heal(image: Image.Image, mask: Image.Image, model, feather: float = 2.0) -> Image.Image:
    """image: RGB; mask: L the same size (white = remove); model: load()'s result. -> the healed RGB picture."""
    desc, device = model
    image = image.convert("RGB")
    mask = mask.convert("L").resize(image.size, Image.Resampling.NEAREST)
    if not (np.asarray(mask) > 127).any():
        raise ValueError("Paint over what you want to remove first.")
    # grow the painted area a little: the edge of a spot or an object usually has a halo that must go too
    grown = mask.point(lambda v: 255 if v > 127 else 0).filter(ImageFilter.MaxFilter(5))
    filled = image.copy()
    for part in regions(np.asarray(grown) > 127):   # each spot close up; later spots see the earlier ones healed
        pm = Image.fromarray(part.astype(np.uint8) * 255)
        x0, y0, x1, y1 = box(part)
        filled.paste(_run(desc, device, filled.crop((x0, y0, x1, y1)), pm.crop((x0, y0, x1, y1))), (x0, y0))
    # only the painted area changes, with a soft edge so it doesn't show
    soft = grown.filter(ImageFilter.GaussianBlur(feather)) if feather else grown
    return Image.composite(filled, image, soft)
