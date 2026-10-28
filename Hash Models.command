#!/bin/bash
# Writes SHA-256 fingerprints of the downloaded model files to model_hashes.txt (read-only).
cd "$(dirname "$0")"
D="$HOME/Library/Application Support/ProtoArt Beautify/downloads"
{ ls -la "$D"; shasum -a 256 "$D"/*; } > model_hashes.txt 2>&1
echo "EXIT CODE: $?" >> model_hashes.txt
echo "Done — see model_hashes.txt"
