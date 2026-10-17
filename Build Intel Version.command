#!/bin/bash
# Builds the Intel-Mac version (dist-intel/) and packages it as dist-intel/ProtoArt-Beautify-1.1-Intel.dmg
cd "$(dirname "$0")"
./build.sh --arch x86_64 2>&1 | tee build-intel.log
STATUS=${PIPESTATUS[0]}
if [[ $STATUS -eq 0 ]]; then
  ./"Make Beautify DMG.command" intel 2>&1 | tee -a build-intel.log
fi
echo "EXIT CODE: $STATUS" | tee -a build-intel.log
