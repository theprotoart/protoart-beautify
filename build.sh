#!/usr/bin/env bash
# Builds "ProtoArt Beautify.app" into ./dist on an Apple Silicon Mac.
#   Needs: macOS 13+, Xcode Command Line Tools (xcode-select --install), internet.
#   Usage: ./build.sh                  build the app (Apple Silicon)
#          ./build.sh --install        build, then copy it to ~/Applications
#          ./build.sh --arch x86_64    build the Intel-Mac version into ./dist-intel (CPU only)
set -euo pipefail

INSTALL=0
ARCH="arm64"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --install) INSTALL=1 ;;
    --arch) ARCH="$2"; shift ;;
    *) echo "unknown option $1"; exit 1 ;;
  esac
  shift
done
[[ "$ARCH" == "arm64" || "$ARCH" == "x86_64" ]] || { echo "--arch must be arm64 or x86_64"; exit 1; }

APP_NAME="ProtoArt Beautify"
BUNDLE_ID="com.protoart.beautify"
VERSION="1.1.0"
PY_VERSION="3.12"

ROOT="$(cd "$(dirname "$0")" && pwd)"
BUILD="$ROOT/.build"
if [[ "$ARCH" == "x86_64" ]]; then
  DIST="$ROOT/dist-intel"; PY_ARCH="x86_64"; REQ="$ROOT/engine/requirements-intel.txt"; PKGS="$BUILD/site-packages-x86_64"
else
  DIST="$ROOT/dist"; PY_ARCH="aarch64"; REQ="$ROOT/engine/requirements.txt"; PKGS="$BUILD/site-packages"
fi
APP="$DIST/$APP_NAME.app"
RES="$APP/Contents/Resources"

say() { printf '\n\033[1;35m▸ %s\033[0m\n' "$*"; }
die() { printf '\n\033[1;31m✗ %s\033[0m\n' "$*" >&2; exit 1; }

[[ "$(uname -s)" == "Darwin" ]] || die "Run this on a Mac."
[[ "$(uname -m)" == "arm64" ]] || die "Build on an Apple Silicon Mac (it can also produce the Intel version with --arch x86_64)."
# /usr/bin/swiftc exists even without the tools (it's a stub that pops up the installer),
# so ask xcrun for the real compiler instead.
if ! xcode-select -p >/dev/null 2>&1 || ! xcrun --find swiftc >/dev/null 2>&1; then
  xcode-select --install >/dev/null 2>&1 || true
  die "Apple's Command Line Tools (the Swift compiler) aren't installed.
   A macOS window should now be asking to install them — click Install, wait for it
   to finish (~5–10 min), then double-click 'Build ProtoArt Beautify.command' again."
fi

mkdir -p "$BUILD" "$DIST"

# ── 1. uv (fast Python installer), kept private to this folder ────────────────
UV="$BUILD/uv/uv"
if [[ ! -x "$UV" ]]; then
  say "Installing uv (private copy in .build/uv)"
  curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="$BUILD/uv" INSTALLER_NO_MODIFY_PATH=1 sh
fi

# ── 2. Standalone, relocatable Python ─────────────────────────────────────────
say "Fetching standalone Python $PY_VERSION"
# --no-bin: don't add python3.x shortcuts to ~/.local/bin (keeps everything inside this folder)
"$UV" python install "$PY_VERSION" --no-bin --install-dir "$BUILD/python"   # the build machine's own (arm64)
# remove shortcuts an earlier build left in ~/.local/bin (only ones pointing into this .build folder)
for l in "$HOME/.local/bin"/python3*; do
  [[ -L "$l" && "$(readlink "$l")" == "$BUILD/"* ]] && rm -f "$l"
done
HOST_PY_DIR="$(find "$BUILD/python" -maxdepth 1 -type d -name "cpython-$PY_VERSION*-macos-aarch64-none" | sort | tail -n1)"
[[ -n "$HOST_PY_DIR" && -x "$HOST_PY_DIR/bin/python3" ]] || die "Couldn't locate the downloaded Python."
HOST_PY="$HOST_PY_DIR/bin/python3"
if [[ "$ARCH" == "x86_64" ]]; then
  say "Fetching standalone Python $PY_VERSION for Intel Macs"
  "$UV" python install "cpython-$PY_VERSION-macos-x86_64-none" --no-bin --install-dir "$BUILD/python"
fi
PY_DIR="$(find "$BUILD/python" -maxdepth 1 -type d -name "cpython-$PY_VERSION*-macos-$PY_ARCH-none" | sort | tail -n1)"
[[ -n "$PY_DIR" && -x "$PY_DIR/bin/python3" ]] || die "Couldn't locate the downloaded $PY_ARCH Python."

# ── 3. App skeleton ───────────────────────────────────────────────────────────
say "Assembling $APP_NAME.app"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$RES"
cp -R "$PY_DIR" "$RES/runtime"
rm -rf "$RES/runtime/include" "$RES/runtime/share" \
       "$RES/runtime/lib/python$PY_VERSION/test" "$RES/runtime/lib/python$PY_VERSION/idlelib" \
       "$RES/runtime/lib/python$PY_VERSION/tkinter" "$RES/runtime/lib/python$PY_VERSION/turtledemo"
rm -f "$RES/runtime/lib/python$PY_VERSION/EXTERNALLY-MANAGED"

# ── 4. Engine dependencies (PyTorch w/ MPS, refiners, spandrel, …) ───────────
# Installed once into .build/site-packages(-x86_64) and reused until the requirements file changes.
STAMP="$PKGS/.requirements.sha"
REQ_SHA="$(shasum "$REQ" | cut -d' ' -f1)-$PY_VERSION-$ARCH"
if [[ ! -f "$STAMP" || "$(cat "$STAMP")" != "$REQ_SHA" ]]; then
  say "Installing engine packages (PyTorch is ~350 MB, this takes a few minutes)"
  rm -rf "$PKGS"
  if [[ "$ARCH" == "x86_64" ]]; then
    # resolve and fetch Intel-Mac wheels from this (arm64) Mac
    "$UV" pip install --python "$HOST_PY" --python-platform x86_64-apple-darwin --target "$PKGS" -r "$REQ"
  else
    "$UV" pip install --python "$PY_DIR/bin/python3" --target "$PKGS" -r "$REQ"
  fi
  # Trim things only needed for compiling extensions.
  rm -rf "$PKGS/torch/include" "$PKGS/torch/share" "$PKGS/torch/test" "$PKGS/bin"
  find "$PKGS" -type d -name "tests" -path "*/numpy/*" -prune -exec rm -rf {} + 2>/dev/null || true
  echo "$REQ_SHA" > "$STAMP"
else
  say "Reusing engine packages from the previous build"
fi
cp -R "$PKGS" "$RES/site-packages"
rm -f "$RES/site-packages/.requirements.sha"

mkdir -p "$RES/engine"
cp "$ROOT"/engine/*.py "$RES/engine/"

# Sanity check: the engine imports (and, on Apple Silicon, sees the GPU). The Intel build is checked
# under Rosetta when it's installed.
RUN=("$RES/runtime/bin/python3")
if [[ "$ARCH" == "x86_64" ]]; then
  if arch -x86_64 /usr/bin/true 2>/dev/null; then RUN=(arch -x86_64 "$RES/runtime/bin/python3")
  else RUN=(); echo "   (Rosetta not installed: skipping the Intel import test)"; fi
fi
if [[ ${#RUN[@]} -gt 0 ]]; then
"${RUN[@]}" -I - "$RES" <<'PY'
import sys, platform; res = sys.argv[1]
sys.path[:0] = [res + "/engine", res + "/site-packages"]
import torch, refiners, spandrel, enhancer  # noqa
print(f"   {platform.machine()} · torch {torch.__version__} · MPS available: {torch.backends.mps.is_available()}")
PY
fi

# ── 4b. Scrub build-machine details and unneeded tools from the bundle ───────
say "Removing build paths and developer tools from the bundle"
BUNDLE_RT="/Applications/$APP_NAME.app/Contents/Resources/runtime"
# command-line tools we never run (their #! lines point at this Mac's build folder)
rm -f "$RES/runtime/bin/"{pip,pip3,pip3.12,idle3,idle3.12,2to3,2to3-3.12,pydoc3,pydoc3.12,python3-config,python3.12-config}
rm -rf "$RES/runtime/lib/pkgconfig"
# text files that record where Python was installed → generic bundle path
{ grep -rlI "$PY_DIR" "$RES/runtime" 2>/dev/null || true; } | while IFS= read -r f; do
  LC_ALL=C sed -i '' "s#$PY_DIR#$BUNDLE_RT#g" "$f"
done
# the shared library's own ID (not used to load it, but it records the build path)
install_name_tool -id "@rpath/libpython$PY_VERSION.dylib" "$RES/runtime/lib/libpython$PY_VERSION.dylib" 2>/dev/null || true
# bytecode: drop caches written during the build, then precompile with bundle-relative paths
find "$RES" -name "__pycache__" -type d -prune -exec rm -rf {} +
say "Pre-compiling Python bytecode (faster launches)"
"$HOST_PY" -I -B -m compileall -q -j 0 -s "$RES" -p "/Applications/$APP_NAME.app/Contents/Resources" \
  "$RES/engine" "$RES/site-packages" "$RES/runtime/lib/python$PY_VERSION" >/dev/null 2>&1 || true
LEAKS="$(grep -rlI "$HOME" "$APP" 2>/dev/null | head -5 || true)"
[[ -z "$LEAKS" ]] && echo "   no personal paths left in the bundle" || { echo "   still mentions $HOME:"; echo "$LEAKS"; }

# ── 5. Native app (SwiftUI) ───────────────────────────────────────────────────
say "Compiling the SwiftUI app"
swiftc -O -parse-as-library -target "$ARCH-apple-macos13.0" \
  -o "$APP/Contents/MacOS/$APP_NAME" "$ROOT/app/ProtoArtBeautify.swift"

# ── 6. Icon ───────────────────────────────────────────────────────────────────
say "Rendering app icon"
ICONSET="$BUILD/AppIcon.iconset"
rm -rf "$ICONSET" && mkdir -p "$ICONSET"
if swift "$ROOT/app/make_icon.swift" "$BUILD/icon_1024.png" >/dev/null; then
  for s in 16 32 128 256 512; do
    sips -z $s $s "$BUILD/icon_1024.png" --out "$ICONSET/icon_${s}x${s}.png" >/dev/null
    d=$((s * 2))
    sips -z $d $d "$BUILD/icon_1024.png" --out "$ICONSET/icon_${s}x${s}@2x.png" >/dev/null
  done
  iconutil -c icns "$ICONSET" -o "$RES/AppIcon.icns" || echo "   (icon skipped)"
else
  echo "   (icon skipped — app still works)"
fi

# ── 7. Info.plist ─────────────────────────────────────────────────────────────
cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>$APP_NAME</string>
  <key>CFBundleDisplayName</key><string>$APP_NAME</string>
  <key>CFBundleIdentifier</key><string>$BUNDLE_ID</string>
  <key>CFBundleExecutable</key><string>$APP_NAME</string>
  <key>CFBundleIconFile</key><string>AppIcon</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>$VERSION</string>
  <key>CFBundleVersion</key><string>$VERSION</string>
  <key>LSMinimumSystemVersion</key><string>13.0</string>
  <key>LSApplicationCategoryType</key><string>public.app-category.photography</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>CFBundleDocumentTypes</key>
  <array>
    <dict>
      <key>CFBundleTypeName</key><string>Image</string>
      <key>CFBundleTypeRole</key><string>Viewer</string>
      <key>LSHandlerRank</key><string>Alternate</string>
      <key>LSItemContentTypes</key><array><string>public.image</string></array>
    </dict>
  </array>
</dict>
</plist>
PLIST

# ── 8. Ad-hoc signature so macOS will launch it ───────────────────────────────
say "Signing (ad-hoc)"
codesign --force --deep --sign - "$APP" >/dev/null 2>&1 || codesign --force --deep --sign - "$APP"

SIZE="$(du -sh "$APP" | cut -f1)"
say "Built: $APP ($SIZE)"
echo "   First launch downloads the models (~5.8 GB download, kept as ~3 GB fp16) to"
echo "   ~/Library/Application Support/$APP_NAME"

if [[ "$INSTALL" == 1 && "$ARCH" == "$(uname -m)" ]]; then
  mkdir -p "$HOME/Applications"
  rm -rf "$HOME/Applications/$APP_NAME.app"
  cp -R "$APP" "$HOME/Applications/"
  say "Installed to ~/Applications/$APP_NAME.app"
fi
