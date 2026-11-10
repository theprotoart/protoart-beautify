"""Long-running engine process for the Mac app.

Protocol (one JSON object per line):
  stdin  -> {"id": "...", "cmd": "enhance", "input": "/path/in", "output": "/path/out.png", "params": {...}}
            {"id": "...", "cmd": "warmup"}      (download + load models)
            {"id": "...", "cmd": "quit"}
  stdout <- {"id": "...", "event": "status", "message": "...", "fraction": 0.42 | null}
            {"id": "...", "event": "done", "output": "/path/out.png", "width": W, "height": H, "seconds": S}
            {"id": "...", "event": "error", "message": "..."}
            {"event": "ready"}  once on startup
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
site = HERE.parent / "site-packages"
if site.is_dir():
    sys.path.insert(0, str(site))

# Let unsupported MPS ops fall back to CPU instead of crashing.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

_out = sys.stdout
sys.stdout = sys.stderr  # keep library prints off the protocol channel


def emit(obj: dict) -> None:
    _out.write(json.dumps(obj) + "\n")
    _out.flush()


_enhancer = None
_paths: dict = {}
_face_fixer = None


def get_enhancer(status):
    global _enhancer
    if _enhancer is None:
        from models import ensure_models
        _paths.update(ensure_models(status))
        from enhancer import Enhancer
        _enhancer = Enhancer(_paths, status)
    return _enhancer


def get_face_fixer():
    global _face_fixer
    from models import DOWNLOADS, _verify
    if _face_fixer is None:
        from face_fix import FaceFixer
        _face_fixer = FaceFixer(dict(_paths), _enhancer.device, _verify)
    for key in ("restoreformer", "codeformer"):  # the app downloads these the first time they're used
        p = DOWNLOADS / f"{key}.pth"
        if p.exists():
            _face_fixer.paths[key] = p
    return _face_fixer


_ICC: list = []


def _srgb_icc() -> bytes | None:
    if not _ICC:
        try:
            from PIL import ImageCms
            _ICC.append(ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes())
        except Exception as e:  # Pillow without LittleCMS: save untagged, as before
            print(f"[tones] no sRGB profile: {e}", flush=True)
            _ICC.append(None)
    return _ICC[0]


def open_image(path: str):
    # The Mac app decodes the user's file with macOS ImageIO and passes a PNG it wrote itself, so only
    # PNG (and, for the command-line tool, JPEG) decoders are enabled here.
    from PIL import Image, ImageOps
    img = Image.open(path, formats=["PNG", "JPEG"])
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        return bg
    return img.convert("RGB")


def handle(req: dict) -> None:
    rid = req.get("id")
    last = [0.0]

    def status(message: str, fraction: float | None, eta: float | None = None) -> None:
        now = time.monotonic()
        # Throttle chatty progress, but always pass through phase changes.
        if fraction is not None and now - last[0] < 0.1 and fraction < 1.0:
            return
        last[0] = now
        emit({"id": rid, "event": "status", "message": message, "fraction": fraction,
              "eta": None if eta is None else round(eta, 1)})

    cmd = req.get("cmd")
    if cmd == "warmup":
        get_enhancer(status)
        emit({"id": rid, "event": "done"})
    elif cmd == "enhance":
        enh = get_enhancer(status)  # first call loads models (not counted in "seconds")
        t0 = time.time()
        image = open_image(req["input"])
        params = dict(req.get("params") or {})
        face = params.pop("face_fix", None)  # {"model", "strength", "fidelity", "faces": [[[x, y] * 5], ...]}
        detail = params.pop("face_detail", None)  # {"strength", "hair", "faces": [[[x, y] * 5], ...]}
        keep = bool(params.pop("keep_tones", True))
        protect = params.pop("protect_faces", None)  # {"faces": [[[x, y] * 5], ...]}: faces skip the AI re-detailing
        mask_path = params.pop("protect_mask", None)  # painted areas (white = keep), a PNG the app made
        detail_mode = params.get("mode", "detail") == "detail"
        protecting = isinstance(protect, dict) and bool(protect.get("faces")) and detail_mode
        painted = None
        result = enh.enhance(image, status, **params)
        from tones import keep_tones, mean_luma
        lum_in, lum_ai = mean_luma(image), mean_luma(result)
        if keep:  # the AI's new detail, with the photo's own light and colour
            status("Keeping the photo's tones", None)
            result = keep_tones(result, image)
        print(f"[tones] brightness in={lum_in:.1f} ai={lum_ai:.1f} out={mean_luma(result):.1f} "
              f"keep_tones={'on' if keep else 'off'}", flush=True)
        faces_detailed = 0
        faces_fixed = 0
        if detail_mode and isinstance(mask_path, str) and mask_path:
            try:  # 🖌 painted areas, like Protoart Studio's Protect from Enhance brush
                from PIL import Image, ImageFilter
                pm = Image.open(mask_path, formats=["PNG"]).convert("L").resize(result.size, Image.Resampling.BILINEAR)
                if pm.getbbox():
                    painted = pm.filter(ImageFilter.GaussianBlur(max(2.0, min(result.size) / 250)))
            except Exception as e:
                print(f"[protect] painted mask skipped: {e}", flush=True)
        if protecting or painted is not None:
            # Protect Faces (like Protoart Studio's Protect from Enhance, found automatically): the faces come from a
            # clean upscale of the photo, with Face Fix (CodeFormer / RestoreFormer) applied to it first; the rest of
            # the picture keeps the AI's new detail. Face Detail and the usual after-enhance Face Fix are skipped.
            from PIL import Image, ImageChops
            from facecrop import boxes_from_landmarks, protect_mask
            status("Protect: clean upscale for the protected areas", None)
            base = enh.enhance(image, status, **dict(params, mode="instant"))
            if base.size != result.size:
                base = base.resize(result.size, Image.Resampling.LANCZOS)
            if protecting and isinstance(face, dict):
                base, faces_fixed = get_face_fixer().fix(
                    base, protect["faces"], model=str(face.get("model", "codeformer")),
                    strength=float(face.get("strength", 0.8)), fidelity=float(face.get("fidelity", 0.5)),
                    status=lambda m, f: status(m, f))
            mask = Image.new("L", result.size, 0)
            if protecting:
                boxes = boxes_from_landmarks(protect["faces"], result.width, result.height, min_eye_px=2)
                mask = protect_mask(result.size, boxes)
            if painted is not None:
                mask = ImageChops.lighter(mask, painted)
            result = Image.composite(base.convert(result.mode), result, mask)
            print(f"[protect] faces={len(protect['faces']) if protecting else 0} painted={painted is not None} "
                  f"face fix first: {protecting and face is not None} ({faces_fixed} fixed)", flush=True)
            if protecting:
                detail = None
                face = None
        if isinstance(detail, dict) and detail.get("faces") and params.get("mode", "detail") == "detail":
            from facecrop import boxes_from_landmarks
            boxes = boxes_from_landmarks(detail["faces"], result.width, result.height)[:8]
            if painted is not None:  # faces you painted over stay as they are
                boxes = [b for b in boxes
                         if painted.getpixel((int(min(result.width - 1, max(0, (b[0] + b[2]) / 2))),
                                              int(min(result.height - 1, max(0, (b[1] + b[3]) / 2))))) < 128]
            print(f"[face detail] {len(boxes)} face(s); strength={detail.get('strength')} hair={detail.get('hair')}",
                  flush=True)
            keys = ("prompt", "negative_prompt", "seed", "controlnet_scale", "controlnet_decay", "condition_scale",
                    "num_inference_steps", "solver", "turbo", "profile")
            result, faces_detailed = enh.detail_faces(
                result, boxes, status, strength=float(detail.get("strength", 0.15)), hair=bool(detail.get("hair")),
                **{k: params[k] for k in keys if k in params})
        if face is not None:  # one line in the app's log, to diagnose "no faces found"
            n_in = len(face.get("faces") or []) if isinstance(face, dict) else -1
            print(f"[face fix] {n_in} face(s) from the app; model={face.get('model') if isinstance(face, dict) else '?'} "
                  f"strength={face.get('strength') if isinstance(face, dict) else '?'} result={result.size}", flush=True)
        if isinstance(face, dict) and face.get("faces"):
            result, faces_fixed = get_face_fixer().fix(
                result, face["faces"], model=str(face.get("model", "restoreformer")),
                strength=float(face.get("strength", 0.8)), fidelity=float(face.get("fidelity", 0.5)),
                status=lambda m, f: status(m, f))
        out = Path(req["output"])
        out.parent.mkdir(parents=True, exist_ok=True)
        # Tag the result as sRGB (what the app hands in), so Before and After are shown with the same colours.
        icc = _srgb_icc()
        extra = {"icc_profile": icc} if icc else {}
        if out.suffix.lower() in (".jpg", ".jpeg"):
            result.save(out, format="JPEG", quality=95, subsampling=0, **extra)
        else:
            result.save(out, format="PNG", **extra)
        emit({"id": rid, "event": "done", "output": str(out), "width": result.width,
              "height": result.height, "seconds": round(time.time() - t0, 1),
              "faces_fixed": faces_fixed, "faces_detailed": faces_detailed,
              "faces_protected": len(protect["faces"]) if protecting else 0,
              "areas_protected": painted is not None})
    else:
        emit({"id": rid, "event": "error", "message": f"unknown command {cmd!r}"})


def main() -> None:
    emit({"event": "ready"})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            emit({"event": "error", "message": f"bad request: {e}"})
            continue
        if req.get("cmd") == "quit":
            break
        try:
            handle(req)
        except Exception as e:  # report and keep serving
            traceback.print_exc()
            msg = str(e) or e.__class__.__name__
            chain = repr(e) + repr(e.__cause__) + repr(getattr(e.__cause__, "__cause__", None))
            if "LocalEntryNotFound" in chain or "Errno 9" in chain or "ConnectError" in chain:
                msg = ("Can't reach huggingface.co to download the models. If you use a firewall "
                       "such as Little Snitch or LuLu, allow 'python3' inside ProtoArt Beautify.app "
                       "to connect to huggingface.co and *.hf.co, then reopen the app.")
            if "out of memory" in msg.lower() or "MPS backend out of memory" in msg:
                msg = ("Ran out of memory. Close other apps, use the 8 GB profile, "
                       "or lower the upscale factor. (" + msg.splitlines()[0][:200] + ")")
            emit({"id": req.get("id"), "event": "error", "message": msg})


if __name__ == "__main__":
    main()
