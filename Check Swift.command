#!/bin/bash
# Type-checks the SwiftUI app (fast), output to swiftcheck.log
cd "$(dirname "$0")"
xcrun swiftc -typecheck -parse-as-library -target arm64-apple-macos13.0 app/ProtoArtBeautify.swift > swiftcheck.log 2>&1
echo "EXIT CODE: $?" >> swiftcheck.log
echo "Check finished — see swiftcheck.log"
