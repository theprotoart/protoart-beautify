# Changelog

## 1.2.0 — 2026-10-17

- **Face Fix:** restores faces after the enhance, with two models: **RestoreFormer** (default, natural and faithful)
  and **CodeFormer** (stronger repair, with a Fidelity knob; non-commercial licence). Faces are found with macOS's
  built-in Vision framework, aligned, restored, blended by Strength and pasted back with a feathered mask. Works with
  every preset, including INSTANT. Each face model downloads once, the first time it's used, checked against a pinned
  SHA-256 and loaded in PyTorch's weights-only mode.
- **Smooth zoom:** the mouse wheel (or two-finger scroll) zooms continuously toward the pointer, no fixed steps.
- **HOW TO** is now a large brown button beside the status lights.
- The status line says how many faces were fixed.

## 1.1.0 — 2026-10-16

- **Zoom** in the picture: FIT, 100% (actual pixels), 200%, + / −, pinch; drag to look around. Results load at full
  resolution so you can see the detail.
- **Views:** SINGLE (click to flip before/after), SPLIT (drag anywhere), SIDE by side; all share the same zoom.
- **Adjust panel:** Camera Raw-style Light (Exposure, Contrast, Highlights, Shadows, Whites, Blacks) and Color
  (Temperature, Tint, Vibrance, Saturation); Color Efex-style Bleach Bypass, Dark Contrasts, Detail Extractor and Film
  Grain, restackable, applied before or after the AI, live preview.
- **Presets:** INSTANT and TURBO added; presets no longer change Upscale (use any look at 1×).
- **Saving:** default save folder, PNG / JPEG / TIFF, Copy or Overwrite original; Save (⌘S) and Save As (⇧⌘S).
- **HOW TO** guide inside the app.
- **Speed (8 GB Macs):** low-memory passes, a 768 px square in one tile, cached Turbo add-on.
- **Intel Mac version** (CPU only).
- **Security:** pinned SHA-256 model checks, image decoding by macOS only, libheif removed (CVE-2026-84383), isolated
  engine environment, temp files cleaned, personal paths scrubbed from builds.
- Fixes: before/after slider responds everywhere; panels and help text no longer shift the window.

## 1.0.0

- First version: Finegrain Image Enhancer on the Mac, presets, batch queue, progress and time left.
