# Security policy

## Reporting a vulnerability

Please report security problems privately, not in a public issue:
use **GitHub → Security → Report a vulnerability** (private advisory) on this repository.

Include what you found, how to reproduce it, and the version (in the DMG's file name, or ProtoArt Beautify → About).
You should get a reply within a week. Please give us a reasonable time to fix it before disclosing it publicly.

## Supported versions

Only the latest release gets security fixes. The app has no automatic updates, so check the Releases page.

## How the app is built to be safe

- Everything runs on your Mac. The app has no account, no telemetry and no analytics.
- It only goes online on first launch, to download its models from Hugging Face over HTTPS. Every file is pinned to an
  exact Hugging Face commit, and every file that is used exactly as downloaded is checked against a pinned SHA-256
  fingerprint (`ModelStore.pinnedSHA256` in the app, `PINNED_SHA256` in `engine/models.py`); a file that doesn't
  match is deleted and never loaded.
- Model files load with safe loaders: safetensors, `torch.load(weights_only=True)` and spandrel's restricted
  unpickler for the two older-format files, which are also hash-checked before every load. No remote model code runs.
- Your pictures are decoded only by macOS (ImageIO, kept patched by system updates). The bundled Python engine only
  ever reads a fresh PNG that the app wrote itself, with its decoder limited to PNG/JPEG. (pi-heif/libheif was removed
  because of CVE-2026-84383.)
- The engine runs as an isolated Python process (`python -I`) with a minimal, fixed environment, talking to the app
  only over a private pipe. There are no environment-variable overrides in release builds.
- Temporary copies of your pictures are deleted when the app opens and quits. "Overwrite original" writes a temporary
  file first and swaps it in, and asks before replacing anything.
- Dependency floors cover known issues: PyTorch ≥ 2.6 and Pillow ≥ 12.3 on Apple Silicon. The Intel build must use
  PyTorch 2.2.2 (the last Intel-Mac release); its `torch.load` weakness (CVE-2025-32434) is covered by the SHA-256
  checks above.
- Release builds are checked for personal paths (`build.sh` scrubs the build machine's paths and warns if any remain;
  the DMG script checks again), and every release publishes a SHA-256 checksum for each DMG.
- Builds are ad-hoc signed, not notarised (no Apple Developer ID yet), so macOS asks you to approve the app once.
