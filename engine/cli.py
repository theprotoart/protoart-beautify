"""Command-line use of the engine (handy for testing without the app).

  python cli.py input.jpg output.png [--factor 2] [--profile 8gb] [--seed 42]
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
site = Path(__file__).resolve().parent.parent / "site-packages"
if site.is_dir():
    sys.path.insert(0, str(site))

import os  # noqa: E402

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from server import get_enhancer, open_image  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--factor", type=float, default=2.0)
    ap.add_argument("--profile", default="8gb", choices=["8gb", "16gb"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=18)
    ap.add_argument("--creativity", type=float, default=0.35, help="denoise strength")
    ap.add_argument("--resemblance", type=float, default=0.6, help="controlnet scale")
    ap.add_argument("--prompt", default="masterpiece, best quality, highres")
    ap.add_argument("--max-side", type=int, default=0, help="cap output long side, e.g. 768 (0 = off)")
    a = ap.parse_args()

    def status(msg, frac, eta=None):
        pct = f"{frac * 100:5.1f}% " if frac is not None else "       "
        left = f" · ~{int(eta // 60)}m{int(eta % 60):02d}s left" if eta is not None else ""
        print(f"\r{pct}{msg + left:<80}", end="", file=sys.stderr, flush=True)

    t0 = time.time()
    enh = get_enhancer(status)
    out = enh.enhance(open_image(a.input), status, profile=a.profile, upscale_factor=a.factor,
                      seed=a.seed, num_inference_steps=a.steps, denoise_strength=a.creativity,
                      controlnet_scale=a.resemblance, prompt=a.prompt,
                      max_output_side=a.max_side or None)
    out.save(a.output)
    print(f"\nSaved {a.output} ({out.width}x{out.height}) in {time.time() - t0:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
