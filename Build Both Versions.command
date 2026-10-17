#!/bin/bash
# Rebuilds both versions and their disk images: dist/ (Apple Silicon, also installed) and dist-intel/ (Intel)
cd "$(dirname "$0")"
{
./build.sh --install && ./"Make Beautify DMG.command" && ./build.sh --arch x86_64 && ./"Make Beautify DMG.command" intel
echo "EXIT CODE: $?"
} 2>&1 | tee build-both.log
