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
  exact Hugging Face commit and checked against a pinned SHA-256 fingerprint before it is used (`PINNED_SHA256` in
  `engine/models.py`; the small files are also checked as they download, `ModelStore.pinnedSHA256` in the app). A
  file that doesn't match is deleted and never loaded.
- Model files load with safe loaders: safetensors, `torch.load(weights_only=True)` and spandrel's restricted
  unpickler for the two older-format files, which are also hash-checked before every load. No remote model code runs.
- Your pictures are decoded only by macOS (ImageIO, kept patched by system updates). The bundled Python engine only
  ever reads a fresh PNG that the app wrote itself, with its decoder limited to PNG/JPEG. (pi-heif/libheif was removed
  because of CVE-2026-84383.)
- The engine runs as an isolated Python process (`python -I`) with a minimal, fixed environment, talking to the app
  only over a private pipe. There are no environment-variable overrides in release builds.
- Temporary copies of your pictures are deleted when the app opens and quits. "Overwrite original" writes a temporary
  file first and swaps it in, and asks before replacing anything.
- Builds install exact, reviewed package versions (`engine/constraints*.txt`) and an exact Python release
  (`PY_PATCH` in `build.sh`), so a build can't silently pick up a newer or tampered package.
- Dependency floors cover known issues: PyTorch ≥ 2.6 and Pillow ≥ 13.0 on Apple Silicon. The Intel build must use
  PyTorch 2.2.2 (the last Intel-Mac release); its `torch.load` weakness (CVE-2025-32434) is covered by the SHA-256
  checks above.
- Release builds are checked for personal paths (`build.sh` scrubs the build machine's paths and warns if any remain;
  the DMG script checks again), and every release publishes a SHA-256 checksum for each DMG.
- Builds are ad-hoc signed, not notarised (no Apple Developer ID yet), so macOS asks you to approve the app once.

## Security audit — 14 November 2026 (version 1.3.2)

**What the app is:** a native macOS desktop app (SwiftUI, not Electron) with a bundled Python engine that runs
locally. It has **no** website or server, accounts or logins, payments, database, uploads to anyone, SMS, emails,
analytics or telemetry. It **does** open the user's own image files, save results, download AI model files from
Hugging Face on first use, and run an image AI (Stable Diffusion 1.5 upscaler and face restore) on the Mac. The
public download page on Hugging Face is a single static HTML page with no scripts or forms.

Checked: the code on `main` (commit `20da48e`), all git history and branches, and the published
`ProtoArt-Beautify-1.3.2.dmg` (SHA-256 `eb5ad456…eacd4`, unpacked and scanned). Tests:
`python3 -m unittest discover -s engine/tests -v` (no models, PyTorch or network needed).

### The standard list

| # | Item | Applies? | What we found | Status |
|---|------|----------|---------------|--------|
| 1 | Next.js middleware header bypass | No | No Next.js, no web server, no middleware. | Not applicable |
| 2 | Exposed `.git` folder | Download page only | `.git/config`, `.git/HEAD` and `.env` on the Hugging Face page all return 404. The app ships no `.git`. | OK |
| 3 | Open Supabase storage | No | No Supabase or any cloud storage. | Not applicable |
| 4 | Prices set by the browser | No | Free app, no payments. | Not applicable |
| 5 | Credential stuffing | No | No accounts or logins. | Not applicable |
| 6 | Open database | No | No database. Settings are local macOS preferences. | Not applicable |
| 7 | Polyfill.io / third-party scripts | Download page | No `<script>`, `<iframe>` or `<form>` tags; links only to huggingface.co. The app has no web views. | OK |
| 8 | SMS pumping | No | No SMS or phone numbers. | Not applicable |
| 9 | Prompt injection | No | The optional prompt only describes the picture to an image model. There is no chat AI, and the AI has no tools, network or file access. | Not applicable |
| 10 | Weak JWT secret | No | No tokens or sessions. | Not applicable |
| 11 | Race conditions | Yes (local) | Engine replies are matched to the current job ID (stale results are ignored). Model files are hash-checked then loaded in the same process. Overwrites write a temporary file, then swap it in atomically. | OK |
| 12 | Account pre-hijacking | No | No accounts. | Not applicable |
| 13 | Subdomain takeover | No | No own domains or DNS; only huggingface.co and github.com. | Not applicable |
| 14 | SSRF | No | No server. The app only fetches hard-coded Hugging Face URLs. | Not applicable |
| 15 | Path traversal | Yes | Output names use only the file's name (`lastPathComponent`), so they can't contain `/`. Model files are saved under fixed names. The engine only takes paths from the app over a private pipe. | OK |
| 16 | Password-reset poisoning | No | No passwords or emails. | Not applicable |
| 17 | Source maps | Download page | No JavaScript; `*.map` files return 404. | Not applicable |
| 18 | Clickjacking | No | The page has nothing to click that acts on anything (download links only). The app is not a web page. | Not applicable |
| 19 | Open redirect | No | No web server or redirect handling. | Not applicable |
| 20 | User enumeration | No | No accounts. | Not applicable |

### Risks that fit this kind of app

| Item | Applies? | What we found | Status |
|------|----------|---------------|--------|
| Secrets in the code | Yes | gitleaks: no keys, tokens or passwords. No `.env` files; `.gitignore` blocks them. | OK |
| Secrets in git history | Yes | gitleaks over all 12 commits, every branch and tag: none. Commits use GitHub's private no-reply email. | OK |
| Secrets in the built app | Yes | Published DMG unpacked and scanned: 3 hits, all false alarms inside third-party libraries (a shortened example token in Hugging Face's docs, a setting name in PyTorch). No personal paths (`/Users/…`) or email. | OK |
| Other people's private data | No | No accounts or server; pictures never leave the Mac. Temporary copies live in your private temp folder and are deleted on open and quit. Saved pictures carry no metadata, so GPS location is removed. | OK |
| Logins and permissions | No | No logins. | Not applicable |
| Injection (HTML, code, SQL) | Yes (local) | No web views, no SQL. App ↔ engine messages are JSON built with `JSONSerialization`, never string-joined. The engine runs as `python -I` with a fixed, minimal environment and no shell. | OK |
| Opening your files | Yes | Your files are decoded only by macOS ImageIO. The engine only decodes the PNG/JPEG the app wrote itself; a test proves TIFF, GIF and BMP are refused. Overwriting asks first and swaps in a finished file. | OK |
| Model downloads (integrity) | Yes | The 9 small files were SHA-256-pinned. **The 4 big model files (UNet, text encoder, autoencoder, ControlNet) had no fingerprint**, only HTTPS + pinned commit. Those 4 repositories had also been renamed on Hugging Face, so the app relied on redirects from the old names. | **Fixed:** all 13 files pinned (from Hugging Face's own records; the 9 existing ones re-confirmed). The big files are checked in the background before their one-time conversion; a bad one is deleted and never used. Current repository names. |
| AI model files that can hide code | Yes | `.pth`/`.pt` files load with `torch.load(weights_only=True)` or spandrel's restricted unpickler, after a hash check on every load. The rest are safetensors (data only). No remote model code. | OK |
| What the AI is allowed to do | Yes | Only turn pixels into pixels: no tools, no internet after setup, no files beyond the picture the app hands it. | OK |
| Desktop app hardening | Yes | Native app, not Electron. No custom URL scheme (it only opens image files). App Transport Security is on (HTTPS only). The engine gets a clean environment (no `DYLD_*`, `PYTHON*` or tokens). | OK |
| Code signing | Yes | Ad-hoc signed, not notarised: users approve it once in Privacy & Security. Fixing this needs an Apple Developer ID ($99/year). | Open (recommendation) |
| Dependencies: known issues | Yes | OSV check of every bundled package. Apple Silicon: none. Intel: PyTorch 2.2.2 (the last Intel-Mac release) has known issues. The serious ones are about loading untrusted checkpoints, which the app never does (pinned hashes first); the rest need attacker-controlled input to functions the app doesn't use. | OK (accepted, Intel only) |
| Dependencies: unpinned | Yes | **Most packages were minimums only (`>=`) and Python was "any 3.12"**, so a future build could pull untested or tampered releases. | **Fixed:** exact versions in `engine/constraints.txt` / `constraints-intel.txt` (the ones shipped in 1.3.2, so nothing changes for users) and Python 3.12.15 in `build.sh`. A change of pins triggers a clean reinstall. |
| Build tools | Yes | `uv` is installed once with `curl … \| sh` from astral.sh (latest version), on the build Mac only. | Open (low) |
| Dependabot PR #1 (torch 2.13.0) | Yes | It would set the Intel build to a PyTorch version with no Intel-Mac release, so that build would fail. | Recommend closing |
| Rate limits | No | Nothing costs money and there's no server. Model downloads retry at most 5 times with back-off. | Not applicable |

### Still open (recommendations)

1. **Notarise the app** with an Apple Developer ID, so users don't have to approve it by hand.
2. **Pin the `uv` installer** to a fixed version and consider `--require-hashes` (a hash for every package file).
   Both need PyPI and astral.sh access from the build Mac.
3. **Close Dependabot PR #1** (above). Future security PRs will update `engine/constraints*.txt`: test before merging.

## Added in 1.4.0 (features ported from Protoart Studio)

| Item | What we checked | Status |
|------|-----------------|--------|
| New downloads (8 upscalers, LaMa) | Each file is pinned to an exact version (a Hugging Face commit or a GitHub release file) and to its SHA-256, checked by the app as it downloads and by the engine before every load. The fingerprints were confirmed by downloading each file, and the 5 older `.pth` files were checked to contain only tensor data (no code references). LaMa is safetensors (data only); the common `big-lama.pt` was avoided because spandrel opens `.pt` files with TorchScript, which can run code. | OK |
| Update checker | Off until you say yes (asked once), or when you choose Check for Updates…. It only accepts installers from this repository's own releases on github.com, refuses a release without a `.sha256` file, and checks the download against it before it can be installed. The installer swaps the app atomically and puts the old one back if anything fails. Note: the checksum is published beside the installer, so it guards against broken or swapped downloads, not against someone who controls the GitHub account. | OK |
| Remove spots | The engine only reads the PNGs the app wrote (your photo, decoded by macOS, and your marks); a test proves other formats are refused. The result replaces nothing on disk: it lives in the temporary folder and is deleted when the app opens and quits. | OK |
| Memory tiers | Same pictures, smaller or larger working pieces by this Mac's memory; upscalers retry on the CPU instead of failing. | OK |
| Tests | `python3 -m unittest discover -s engine/tests -v` (23 tests) and the app's `--selftest-features` (17 checks), all passing on an 8 GB Apple Silicon Mac. | OK |

