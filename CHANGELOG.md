# Changelog

## 1.3.1 — 2026-11-09
- **Before on the left, After on the right** in the Split view.
- **Keep Tones** (on by default): the result keeps your photo's own brightness and colour; only the new detail is added (ported from Protoart Studio's "Keep the picture's style"). Results are also tagged sRGB, so Before and After show the same colours.
- **Face Detail** (off by default, 15% strength): each face is redrawn close up by the same AI, guided by its own shapes, colour-matched and blended back (ported from Protoart Studio's face detailer). Optional Hair.
- **Protect** (like Protoart Studio's Protect from Enhance): **Protect Faces** runs Face Fix first on a clean upscale and keeps faces out of the AI re-detailing; **PAINT AREAS** marks anything else to keep as it is, with Studio-style tools: Brush, Erase, Lasso, Polygon, Wand, and Subject / Text found by Apple's built-in Vision (no downloads), Add / Take away, Undo, Clear.
- **Face Fix**: restored faces keep the photo's skin tone; CodeFormer is the default model for new users.
- **Adjust**: Presence (Texture, Clarity, Dehaze, Sharpen, Vignette), Color Mixer, Color Grading, and the Glamour Glow, Blur and Skin Texture filters, ported from Protoart Studio's Adjust. Each has an on/off, a reset and its own guide.
- Checks: gitleaks (no leaks), OSV dependency audit (clean apart from the Intel-only PyTorch 2.2.2 notes in SECURITY.md), filter self-test (all kernels compile; full preview with every filter on in 0.14 s).

## 1.3.0 — 2026-10-24
- **Queue different settings on the same photo**: press **+ QUEUE** (⇧⌘↩) to line up versions; each row shows its settings, and "Use These Settings" brings them back.
- **My Presets**: save, rename, update and delete your own presets.
- **Saves to your Desktop** by default, with the settings in the file name (change the folder any time).
- **Likeness** (IP-Adapter Plus): keeps people and objects looking like themselves while new detail is added. Optional ~1.4 GB download on first use.
- **CodeFormer presets** for Face Fix, and a fix for a "no faces found" case.
- **Bloom**: a Procreate-style glow filter for highlights.
- **Security**: bundled Pillow 13.0.0 (fixes the October 2026 image-decoder advisories), filelock 4.2.5 and setuptools 84.2.1. Secret scan (gitleaks) and dependency audit (OSV) are clean, apart from the Intel-only PyTorch 2.2.2 notes in SECURITY.md.

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
