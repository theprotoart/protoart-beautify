#!/bin/bash
# Double-click to build ProtoArt Beautify. Output is also saved to build.log.
cd "$(dirname "$0")"
./build.sh --install 2>&1 | tee build.log
echo "EXIT CODE: ${PIPESTATUS[0]}" | tee -a build.log
echo
echo "Done — you can close this window."
