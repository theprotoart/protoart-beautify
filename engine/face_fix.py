"""Face fix: restore faces after the enhance pass, with RestoreFormer (default) or CodeFormer.

The Mac app finds the faces with Apple's Vision framework and sends five landmarks per face (both eyes,
nose tip, both mouth corners) as fractions of the image size. Here each face is aligned to the standard
512x512 face template, restored, blended with the original by `strength`, and pasted back through a
feathered oval mask so there are no seams.

No OpenCV or extra detector models are needed: alignment and pasting use PIL's affine transform.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter

# FFHQ 5-point template for a 512 px aligned face (the one GFPGAN / CodeFormer / facexlib use).
TEMPLATE_512 = np.array([
    [192.98138, 239.94708],  # eye on the left of the picture
    [318.90277, 240.19360],  # eye on the right of the picture
    [256.63416, 314.01935],  # nose tip
    [201.26117, 371.41043],  # mouth corner, left of picture
    [313.08905, 371.15118],  # mouth corner, right of picture
], dtype=np.float64)

MIN_EYE_DISTANCE = 12.0  # px in the result; smaller faces are too tiny to restore well


def similarity_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares rotation + uniform scale + translation mapping src points onto dst (Umeyama).

    Returns a 2x3 matrix M with dst ≈ M @ [x, y, 1].
    """
    n = src.shape[0]
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s0, d0 = src - mu_s, dst - mu_d
    cov = d0.T @ s0 / n
    u, sig, vt = np.linalg.svd(cov)
    d = np.ones(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        d[-1] = -1
    r = u @ np.diag(d) @ vt
    var_s = (s0 ** 2).sum() / n
    scale = (sig * d).sum() / var_s
    t = mu_d - scale * r @ mu_s
    return np.hstack([scale * r, t[:, None]])


def invert_affine(m: np.ndarray) -> np.ndarray:
    a = np.vstack([m, [0.0, 0.0, 1.0]])
    return np.linalg.inv(a)[:2]


def _pil_coeffs(m: np.ndarray) -> tuple[float, ...]:
    """PIL's AFFINE transform maps *output* pixels to *input* pixels: pass that mapping's 2x3 matrix."""
    return tuple(float(v) for v in m.reshape(-1))


def _face_mask(size: int = 512) -> Image.Image:
    """Feathered oval covering the face and forehead in the aligned 512 crop."""
    m = Image.new("L", (size, size), 0)
    k = size / 512
    ImageDraw.Draw(m).ellipse([72 * k, 44 * k, 440 * k, 458 * k], fill=255)
    return m.filter(ImageFilter.GaussianBlur(18 * k))


class FaceFixer:
    def __init__(self, paths: dict[str, Path], device: torch.device, verify: Callable[[str, Path], None]) -> None:
        self.paths = paths
        # Small model, one face at a time: MPS in float32 on Apple Silicon, otherwise the CPU.
        self.device = device if device.type in ("mps", "cuda") else torch.device("cpu")
        self.verify = verify
        self.models: dict[str, Any] = {}
        self.mask = _face_mask()

    def _model(self, kind: str) -> Any:
        if kind in self.models:
            return self.models[kind]
        key = "codeformer" if kind == "codeformer" else "restoreformer"
        path = self.paths.get(key)
        if path is None or not Path(path).exists():
            raise RuntimeError(f"The {kind.title()} face model isn't downloaded yet. Quit and reopen the app "
                               "to download it.")
        self.verify(key, Path(path))  # pinned SHA-256 before any unpickling
        # weights_only: only tensors and plain containers can be unpickled (no code), on every torch version.
        state = torch.load(str(path), map_location="cpu", weights_only=True)
        state = state.get("params_ema", state.get("params", state))
        if kind == "codeformer":
            from codeformer_arch import CodeFormer
            net = CodeFormer(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
                             connect_list=["32", "64", "128", "256"])
        else:
            from restoreformer_arch import RestoreFormer
            net = RestoreFormer()
        net.load_state_dict(state, strict=True)
        del state
        net.eval().to(self.device)
        # Keep only one face model in memory at a time (8 GB Macs).
        self.models.clear()
        self.models[kind] = net
        return net

    @torch.no_grad()
    def _restore_crop(self, net: Any, kind: str, crop: Image.Image, fidelity: float) -> Image.Image:
        x = torch.from_numpy(np.asarray(crop, dtype=np.float32) / 255.0).permute(2, 0, 1)
        x = ((x - 0.5) / 0.5).unsqueeze(0).to(self.device)
        if kind == "codeformer":
            y = net(x, w=float(fidelity), adain=True)[0]
        else:
            y = net(x)
            y = y[0] if isinstance(y, (tuple, list)) else y
        y = y.squeeze(0).float().clamp(-1, 1).cpu()
        y = ((y + 1) / 2 * 255.0).round().byte().permute(1, 2, 0).numpy()
        return Image.fromarray(y, "RGB")

    def fix(self, image: Image.Image, faces: list, *, model: str = "restoreformer", strength: float = 0.8,
            fidelity: float = 0.5, status: Callable[[str, float | None], None] | None = None) -> tuple[Image.Image, int]:
        """Return (image with faces restored, number of faces restored)."""
        kind = "codeformer" if model == "codeformer" else "restoreformer"
        strength = float(max(0.0, min(1.0, strength)))
        fidelity = float(max(0.0, min(1.0, fidelity)))
        if not faces:
            return image, 0
        strength = max(0.1, strength)  # 0 would skip every face; the app's knob starts at 10%
        img = image.convert("RGB")
        w, h = img.size
        todo = []
        for f in faces:
            try:
                pts = np.array(f, dtype=np.float64).reshape(5, 2) * [w, h]
            except (ValueError, TypeError):
                continue
            if not np.all(np.isfinite(pts)) or np.linalg.norm(pts[1] - pts[0]) < MIN_EYE_DISTANCE:
                continue
            todo.append(pts)
        if not todo:
            return image, 0

        net = self._model(kind)
        for n, pts in enumerate(todo):
            if status:
                status(f"Fixing faces with {kind.title()} — {n + 1}/{len(todo)}", n / len(todo))
            m = similarity_transform(pts, TEMPLATE_512)       # picture -> aligned 512 crop
            inv = invert_affine(m)                            # aligned crop -> picture
            crop = img.transform((512, 512), Image.Transform.AFFINE, _pil_coeffs(inv),
                                 resample=Image.Resampling.BICUBIC)
            fixed = self._restore_crop(net, kind, crop, fidelity)
            if strength < 1:
                fixed = Image.blend(crop, fixed, strength)

            # Paste back only over the face's bounding box (cheap even on very large results).
            corners = np.array([[0, 0, 1], [512, 0, 1], [0, 512, 1], [512, 512, 1]], dtype=np.float64) @ inv.T
            x0, y0 = np.floor(corners.min(0)).astype(int) - 2
            x1, y1 = np.ceil(corners.max(0)).astype(int) + 2
            x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
            if x1 <= x0 or y1 <= y0:
                continue
            # Output pixel (u, v) in the box -> picture (u + x0, v + y0) -> crop via m.
            shift = m @ np.array([[1, 0, x0], [0, 1, y0], [0, 0, 1]], dtype=np.float64)
            coeffs = _pil_coeffs(shift)
            size = (x1 - x0, y1 - y0)
            face_back = fixed.transform(size, Image.Transform.AFFINE, coeffs, resample=Image.Resampling.BICUBIC)
            mask_back = self.mask.transform(size, Image.Transform.AFFINE, coeffs, resample=Image.Resampling.BILINEAR)
            region = img.crop((x0, y0, x1, y1))
            region.paste(face_back, (0, 0), mask_back)
            img.paste(region, (x0, y0))

        if self.device.type == "mps":
            torch.mps.empty_cache()
        if status:
            status("Faces fixed", 1.0)
        return img, len(todo)
