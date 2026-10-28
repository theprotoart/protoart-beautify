#!/bin/bash
# Installs gitleaks with Homebrew if needed, then scans the ProtoArt Beautify source (writes gitleaks-scan.log).
cd "$(dirname "$0")"
LOG="$PWD/gitleaks-scan.log"
{
  BREW="$(command -v brew || ls /opt/homebrew/bin/brew /usr/local/bin/brew 2>/dev/null | head -1)"
  GL="$(command -v gitleaks || ls /opt/homebrew/bin/gitleaks /usr/local/bin/gitleaks 2>/dev/null | head -1)"
  if [[ -z "$GL" ]]; then
    if [[ -z "$BREW" ]]; then echo "HOMEBREW NOT FOUND"; echo "GITLEAKS EXIT: none"; exit 0; fi
    echo "Installing gitleaks with $BREW"
    HOMEBREW_NO_AUTO_UPDATE=1 "$BREW" install gitleaks
    GL="$(dirname "$BREW")/gitleaks"
  fi
  T="$(mktemp -d)"; tar xzf pab-repo.tgz -C "$T"
  "$GL" version
  "$GL" dir "$T/pab-repo" --no-banner --redact -v
  echo "GITLEAKS EXIT: $?"
  rm -rf "$T"
} > "$LOG" 2>&1
echo "Done — see gitleaks-scan.log"
