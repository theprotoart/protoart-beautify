"""Face close-ups for Face Detail (ported from Protoart Studio's detailer.py: face_crop / paste_back)."""

from __future__ import annotations

from PIL import Image, ImageDraw, ImageFilter


def boxes_from_landmarks(faces: list, width: int, height: int, min_eye_px: float = 6.0) -> list[tuple]:
    """The app's five landmarks per face (fractions of the picture: both eyes, nose, mouth corners) -> face boxes
    (x1, y1, x2, y2) in pixels, sized like a face detector's box (about 2.4 eye-distances wide), biggest first."""
    out = []
    for f in faces or []:
        try:
            pts = [(float(x) * width, float(y) * height) for x, y in f][:5]
        except (TypeError, ValueError):
            continue
        if len(pts) < 5:
            continue
        (lx, ly), (rx, ry) = pts[0], pts[1]
        eye = ((rx - lx) ** 2 + (ry - ly) ** 2) ** 0.5
        if not eye >= min_eye_px:  # also rejects NaN
            continue
        mouth_y = (pts[3][1] + pts[4][1]) / 2
        eyes_y = (ly + ry) / 2
        cx = sum(p[0] for p in pts) / 5
        cy = (eyes_y + mouth_y) / 2
        side = max(2.4 * eye, 2.0 * abs(mouth_y - eyes_y) * 1.6)
        out.append((cx - side / 2, cy - side * 0.55, cx + side / 2, cy + side * 0.45))
    out.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    return out


def face_crop(image: Image.Image, box: tuple, padding: float = 1.8, work: int = 512, grow: float = 0.18,
              blur: float = 1.0, hair: bool = False) -> tuple[Image.Image, tuple, Image.Image]:
    """Square region around a face (resized to work x work), its box in the picture, and the soft oval mask
    in work-size coordinates. hair: the mask also covers the hair above and beside the face."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * padding
    side = min(side, max(image.width, image.height))
    left = int(max(0, min(image.width - side, cx - side / 2)))
    top = int(max(0, min(image.height - side, cy - side / 2)))
    side_i = max(8, int(min(side, image.width - left, image.height - top)))
    region = (left, top, left + side_i, top + side_i)
    crop = image.crop(region).resize((work, work), Image.Resampling.LANCZOS)
    s = work / side_i
    fx1, fy1, fx2, fy2 = (x1 - left) * s, (y1 - top) * s, (x2 - left) * s, (y2 - top) * s
    g = max(0.0, min(1.0, float(grow))) * (fx2 - fx1)
    mask = Image.new("L", (work, work), 0)
    if hair:
        fw = fx2 - fx1
        ImageDraw.Draw(mask).ellipse([fx1 - g - 0.25 * fw, fy1 - g - 0.55 * fw, fx2 + g + 0.25 * fw, fy2 + g], fill=255)
    else:
        ImageDraw.Draw(mask).ellipse([fx1 - g, fy1 - g * 1.4, fx2 + g, fy2 + g], fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(max(0.5, work / 48 * max(0.1, min(4.0, float(blur))))))
    return crop, region, mask


def paste_back(image: Image.Image, new_crop: Image.Image, region: tuple, mask: Image.Image) -> Image.Image:
    """Blend the redrawn face back through the soft mask."""
    left, top, right, bottom = region
    size = (right - left, bottom - top)
    patch = new_crop.convert(image.mode).resize(size, Image.Resampling.LANCZOS)
    m = mask.resize(size, Image.Resampling.LANCZOS)
    out = image.copy()
    out.paste(patch, (left, top), m)
    return out
