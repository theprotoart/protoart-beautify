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


def get_enhancer(status):
    global _enhancer
    if _enhancer is None:
        from models import ensure_models
        paths = ensure_models(status)
        from enhancer import Enhancer
        _enhancer = Enhancer(paths, status)
    return _enhancer


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
        result = enh.enhance(image, status, **(req.get("params") or {}))
        out = Path(req["output"])
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.suffix.lower() in (".jpg", ".jpeg"):
            result.save(out, format="JPEG", quality=95, subsampling=0)
        else:
            result.save(out, format="PNG")
        emit({"id": rid, "event": "done", "output": str(out), "width": result.width,
              "height": result.height, "seconds": round(time.time() - t0, 1)})
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
