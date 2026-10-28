#!/bin/bash
# Re-makes both disk images (Apple Silicon from dist/, then rebuilds and packages the Intel version)
cd "$(dirname "$0")"
{
./"Make Beautify DMG.command" && ./build.sh --arch x86_64 && ./"Make Beautify DMG.command" intel
echo "EXIT CODE: $?"
} 2>&1 | tee build-dmgs.log
