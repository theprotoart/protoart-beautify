#!/bin/bash
# Packages the app into a disk image to share: dist/ProtoArt-Beautify-1.2.dmg
# ("intel" argument: dist-intel/ProtoArt-Beautify-1.2-Intel.dmg)
cd "$(dirname "$0")"
LOGF="dmg.log"; [[ "${1:-}" == "intel" ]] && LOGF="dmg-intel.log"
exec > >(tee "$LOGF") 2>&1
set -euo pipefail
if [[ "${1:-}" == "intel" ]]; then DIST="dist-intel"; SUFFIX="-Intel"; LOG="dmg-intel.log"; else DIST="dist"; SUFFIX=""; LOG="dmg.log"; fi
APP="$DIST/ProtoArt Beautify.app"
[[ -d "$APP" ]] || { echo "Build the app first (Build ProtoArt Beautify.command)"; exit 1; }

echo "▸ Checking the app's signature"
codesign --verify --deep --strict "$APP"

echo "▸ Checking the bundle for personal paths"
if grep -rlI --exclude-dir=__pycache__ "$HOME" "$APP" 2>/dev/null | head -5 | grep .; then
  echo "  (warning: the files above mention $HOME)"
else
  echo "  none found"
fi

STAGE="$(mktemp -d)/ProtoArt Beautify"
mkdir -p "$STAGE"
ditto "$APP" "$STAGE/ProtoArt Beautify.app"
xattr -cr "$STAGE/ProtoArt Beautify.app"
ln -s /Applications "$STAGE/Applications"
cp "Read Me First.txt" "$STAGE/"
[[ -n "$SUFFIX" ]] && cp "Read Me First (Intel).txt" "$STAGE/" 2>/dev/null || true

OUT="$DIST/ProtoArt-Beautify-1.2$SUFFIX.dmg"
rm -f "$OUT"
# a previously opened copy mounted under the same name makes hdiutil fail: eject it first
for v in "/Volumes/ProtoArt Beautify"*; do [[ -d "$v" ]] && hdiutil detach "$v" -quiet -force 2>/dev/null || true; done
echo "▸ Creating $OUT (compressing ~800 MB, takes a few minutes)"
hdiutil create -volname "ProtoArt Beautify" -srcfolder "$STAGE" -fs HFS+ -format UDZO \
  -imagekey zlib-level=9 "$OUT" >/dev/null
rm -rf "$(dirname "$STAGE")"
hdiutil verify "$OUT" >/dev/null && echo "  image verified"
shasum -a 256 "$OUT" | tee "$OUT.sha256"
du -h "$OUT"
echo "EXIT CODE: 0"
