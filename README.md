# ProtoArt Beautify

A local AI image enhancer for Macs, styled like a vintage studio rack unit. It recreates the
[Finegrain Image Enhancer](https://huggingface.co/spaces/finegrain/finegrain-image-enhancer) on your own Mac, tuned to
run on an **8 GB** Mac: no account, no cloud, no telemetry.

Drop in an image → it's pre-upscaled 4× with ESRGAN, resized to your target size, then re-detailed tile by tile with
Stable Diffusion 1.5 (Juggernaut Reborn) guided by a Tile ControlNet. Compare before/after (single, split or side by
side, with smooth mouse-wheel zoom), fix faces with **RestoreFormer** or **CodeFormer**, finish it with Camera
Raw-style Light/Color sliders and Color Efex-style filters, and save.

> **Models are not included.** The app downloads its models (~6 GB, once) from Hugging Face on first launch, and
> **each model has its own licence**; some are non-commercial. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Download

- **[GitHub Releases](https://github.com/theprotoart/protoart-beautify/releases/latest)**
  - `ProtoArt-Beautify-<version>.dmg`: Apple Silicon Macs (M1 or newer)
  - `ProtoArt-Beautify-<version>-Intel.dmg`: Intel Macs (runs on the processor, much slower)
- **Hugging Face:** [huggingface.co/spaces/brushpenbob/protoart-beautify](https://huggingface.co/spaces/brushpenbob/protoart-beautify)

**Requirements:** macOS 13 Ventura or newer, ~10 GB free disk space, internet for the first launch. 8 GB of memory
works; 16 GB is more comfortable.

**Install:** open the DMG and drag the app to Applications. Test builds are not notarised, so on first launch use
**System Settings → Privacy & Security → Open Anyway** (once). Check your download with
`shasum -a 256 ProtoArt-Beautify-<version>.dmg` against the `.sha256` file on the release.

**Also by me:** [Protoart Studio](https://github.com/theprotoart/protoart-studio), a local AI image studio.

## Using it

The app is styled as a studio rack unit: black anodized faceplate with procedural
scratches and wear (generated at launch, no image assets), rack screws, bakelite knobs,
bat-handle toggles, status LEDs and an analog VU meter.

- **Knobs:** drag up or right to turn clockwise; the amber window shows the value.
  **Max Size** is a detented selector: click to step through 768 / 1K / 1.5K / 2K / ∞, or drag.
- **Toggles:** click to flip (Batch, Memory 8 GB/16 GB+, Save mode Copy/Overwrite, Solver DDIM/DPM).
- **Prompts** are written on the masking-tape strips; click to edit.
- **LEDs:** READY (green), PROCESS/LOADING (amber, blinking), FAULT (red).
- **Meter bridge:** the VU needle tracks progress; readouts show percent, time left and,
  in batch mode, image *n/N* and time left for the whole queue.

- **Presets:** INSTANT (4× sharpening upscaler only, ~10 s, invents nothing), TURBO (768 px, distilled
  LCM add-on: 2 quick passes, about twice as fast as Fast), FAST (768 px, fewer smarter steps), BALANCED (the Space's settings, 1024 px),
  QUALITY (more steps, 1536 px), SUBTLE (gentle cleanup, stays faithful) and CREATIVE (adds lots
  of new detail). The lit key shows which preset matches; turn any knob and it becomes custom.
- **Guide strip:** hover over any knob, switch or preset and the amber strip under the monitor
  explains in plain language what it does.

- **Detail filters** (Nik Color Efex-style): **Bleach Bypass**, **Dark Contrasts** and **Detail Extractor**,
  in a Color Efex-style **Adjust** panel left of the picture (toggle it with the **ADJUST** key next to
  the view keys). The panel starts with Camera Raw-style **Light** (Exposure, Contrast, Highlights, Shadows,
  Whites, Blacks) and **Color** (Temperature, Tint, Vibrance, Saturation) sliders; double-click to reset.
  Filters: Bleach Bypass, Dark Contrasts, Detail Extractor and **Film Grain**, applied top to bottom;
  restack them with the ▲▼ buttons. Click a filter's name to apply it and open its knobs; the checkbox turns it on or off;
  knobs only adjust. Live preview (Core Image on the GPU). **Apply: AFTER AI** puts them on top of
  the result (adjust any time); **BEFORE AI** filters the photo first and the AI enhances that. They also
  work on their own, with no enhancement.
- **View keys** (meter bridge): **SINGLE** (click the picture to flip to the original), **SPLIT**
  (drag anywhere to move the divider), **SIDE** (side by side).
- **Save module**: default **Save To** folder, **Format** PNG / JPEG / TIFF, and **Mode** COPY (a brand-new
  file, original untouched) or OVERWRITE (replaces the original, same name and format, asks first).
  **SAVE** (⌘S) uses these; **SAVE AS** (⇧⌘S) picks any name, folder and format. Filters are always included.
- **HOW TO** (⌘?): a built-in guide to every control plus fixes for common problems.
- **Single mode** (default): drop an image, press the light-blue **Enhance** button (⌘↩), drag the before/after
  slider, then **Save As** (PNG/JPEG/TIFF).
- **Batch mode** (toolbar toggle or ⌘B): a queue panel appears. Drop many images or whole
  folders, press **Enhance N**. Results auto-save with the Save module's settings
  (default `~/Pictures/ProtoArt Beautify`, named `photo-beautify.png`, or over the originals in OVERWRITE mode).
  You can keep adding images while it runs. Click the ✕ on any queued or running row to
  cancel just that image (cancelling the running one moves straight on to the next);
  right-click a row to cancel, re-run, remove or show it in Finder. **Stop** (⌘.) halts
  the whole queue. The Dock icon shows how many are left
  and a chime plays when the batch finishes.
- **Progress:** every image shows a live percentage and time remaining; in batch mode the
  status bar also shows *Image 2 of 5* and an estimate for the whole queue. Estimates come
  from timing each stage (ESRGAN tiles, VAE, diffusion passes); the very first image shows
  its estimate once detail generation starts, later ones from the start.
- **Max size** caps the long side of the result. Batch mode defaults to **768 px** (fastest,
  lightest on memory); single mode defaults to no limit. Each mode remembers its own setting.
  Large inputs are shrunk first so a 2× detail pass lands exactly on the cap.

## Build

Requirements: Apple Silicon Mac, macOS 13+, Xcode Command Line Tools (`xcode-select --install`), internet.

```bash
cd ProtoArtBeautify
./build.sh            # → dist/ProtoArt Beautify.app
./build.sh --install  # same, then copies it to ~/Applications
```

The build fetches a private Python 3.12 + PyTorch into the app bundle (nothing is
installed system-wide). The app is ~1 GB, mostly PyTorch.

**First launch** downloads the models (~5.8 GB) and converts them once to fp16
(~3 GB kept) in `~/Library/Application Support/ProtoArt Beautify`. After that it runs offline.

## How it maps to the Space

| Space | ProtoArt Beautify |
| --- | --- |
| Juggernaut Reborn SD1.5 UNet / text encoder / VAE, Tile ControlNet, 4x-UltraSharp, JuggernautNegative embedding, `more_details` (0.5) + `SDXLrender_v2` (1.0) LoRAs — pinned revisions | Same files, same revisions |
| refiners `MultiUpscaler` @ `cfe8b66` | Same library and commit |
| Defaults: prompt, negative, seed 42, 2×, ControlNet 0.6, decay 1.0, CFG 6, denoise 0.35, 18 steps, DDIM | Same defaults |
| CUDA, bf16/fp32 | Apple GPU (MPS), fp16 |

Slider names in the app: **Creativity** = denoise strength, **Resemblance** = ControlNet scale, **HDR** = condition (CFG) scale.

## 8 GB tuning

The **Memory** picker sets a profile (auto-selected from your RAM):

| | 8 GB (safe) | 16 GB+ (Space defaults) |
| --- | --- | --- |
| Input short side capped at | 512 px | 768 px |
| Diffusion tile (latent) | 64×64 (512 px) | 144×112 (1152×896 px) |
| VAE tile | 512 px | 1024 px |
| ESRGAN tile | 192 px | 256 px |
| Max output | 8 MP | 16 MP |

Other things that keep memory down:
- All weights stored and loaded as fp16 (≈2.9 GB resident), converted tensor-by-tensor so there's never a full fp32 copy in RAM.
- 512 px diffusion tiles keep each attention map around 0.3 GB instead of several GB.
- Text encoder is offloaded after each prompt; caches are flushed between phases.
- Models stay loaded between runs, so only the first enhance pays the load cost.

Measured on an 8 GB M1 (608×768 result): **Instant** ≈ 9 s, **Turbo** ≈ 50 s, **Fast** ≈ 1–2 min
(it swings with how much memory other apps use), **Balanced**
(1024 px) ≈ 4½ min, plus ~25 s to load the models once per launch. Exact times appear in the
app after the first image.

Speed work in the engine (each measured on that M1):
- **Low-memory passes** (always on): the GPU cache goes back to macOS after every diffusion pass,
  with 512-token attention chunks. Identical output; helps most when the Mac is close to swapping
  (one run: 121 s → 62 s; under heavier memory pressure both took ~125 s).
- **Sliced attention** (1024-token chunks): stopped a swap stall where Fast never finished in 13 min.
- **Tile planner + /64 padding**: one whole-image tile at 768 px instead of four overlapping
  512 px tiles; fewest tiles within a 9216-token budget (a 768 px square is one tile) for bigger outputs. Fast 182 s → 92 s.
- **LoRAs merged into the weights** at load (350 layers) instead of running as extra layers.
- **DPM / LCM solvers built on the CPU** first (they need float64, which Apple GPUs lack).
- **Turbo (LCM-LoRA)**: 51 s vs ~125 s for Fast in the same session, similar look. Its layer
  match is cached next to the file, and it loads with the other models (~35 s launch total).
  `PAB_TURBO=0` skips it.

Run only one Stable Diffusion app at a time on 8 GB: two engines (~3 GB of models each) push
the Mac into swap and everything slows down several-fold.
Close memory-heavy apps (browsers with many tabs) for the smoothest runs.

## Security & privacy

- Images never leave the Mac; no analytics. Network is used only for the one-time model download
  (HTTPS, every file pinned to an exact Hugging Face commit).
- Untrusted image files are decoded only by macOS ImageIO (patched by system updates); the engine
  receives a fresh sRGB PNG written by the app, and its Pillow decoder is restricted to PNG/JPEG.
  pi-heif/libheif was removed (libheif 1.23.0 is affected by CVE-2026-84383).
- Pickle-format model files load in safe mode (`torch.load(weights_only=True)`, spandrel's
  restricted unpickler); dependency floors: torch ≥ 2.6, Pillow ≥ 12.3.
- The engine runs isolated (`python -I`) with a minimal environment; no environment overrides.
- Temporary copies of images are deleted at launch and quit; Overwrite writes a temp file then swaps it in.
- Ad-hoc signed (not notarized): recipients approve it once in System Settings › Privacy & Security.

## Sharing

`Make Beautify DMG.command` packages `dist/ProtoArt Beautify.app` with an Applications link and
`Read Me First.txt` into `dist/ProtoArt-Beautify-1.2.dmg` (+ SHA-256).

**Intel Macs:** `Build Intel Version.command` (or `./build.sh --arch x86_64`) builds an x86_64 copy into
`dist-intel/` from this Apple Silicon Mac and packages `ProtoArt-Beautify-1.2-Intel.dmg`. It runs the AI
on the CPU (PyTorch 2.2.2, the last Intel-Mac release; float32), starts on TURBO, and is much slower:
INSTANT took ~2 min at 1024 px under Rosetta. `Build Both Versions.command` rebuilds both DMGs.
PyTorch 2.2.2's `torch.load` weakness (CVE-2025-32434) is covered by SHA-256-pinning every pickle file
before it is loaded (engine/models.py; the app also checks pinned hashes right after download).

## Troubleshooting

- **Engine log:** `~/Library/Logs/ProtoArt Beautify.log` (also via *Show Log* on any error).
- **"Ran out of memory" / engine stopped:** use the 8 GB profile, lower Upscale, close other apps.
- **Command-line test without the UI:**
  ```bash
  A="dist/ProtoArt Beautify.app/Contents/Resources"
  "$A/runtime/bin/python3" -I "$A/engine/cli.py" photo.jpg out.png --factor 2 --max-side 768
  ```
- **Reset models:** delete `~/Library/Application Support/ProtoArt Beautify`.

## Layout

```
SECURITY.md · PRIVACY.md · THIRD_PARTY_NOTICES.md · CHANGELOG.md · RELEASING.md
app/ProtoArtBeautify.swift   SwiftUI app (queue, before/after slider, settings, progress, save)
app/make_icon.swift          renders the app icon at build time
engine/server.py             long-running engine; JSON lines over stdin/stdout
engine/enhancer.py           refiners MultiUpscaler + memory profiles + size caps
engine/progress.py           per-stage timing, percentage and time-remaining estimates
engine/esrgan.py             4x-UltraSharp via spandrel, feathered tiling
engine/models.py             pinned downloads + one-time fp16 conversion
engine/cli.py                command-line entry point
build.sh                     builds dist/ProtoArt Beautify.app
```

Model licenses: Juggernaut Reborn and the ControlNet follow the CreativeML OpenRAIL-M
license; 4x-UltraSharp is CC BY-NC-SA 4.0 (non-commercial). Check them before commercial use.
