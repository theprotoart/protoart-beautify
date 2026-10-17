# Releasing ProtoArt Beautify

Releases are built on an Apple Silicon Mac and published as GitHub Releases. The repository holds source code only:
**never commit the `.app`, the `.dmg`, model weights, pictures or any key / token.**

## 1. Prepare

1. Update `VERSION` in `build.sh` and the DMG name in `Make Beautify DMG.command`; add a section to `CHANGELOG.md`.
2. Make sure the working tree is clean: `git status`.
3. Scan for secrets before every push: `gitleaks dir .` (or `gitleaks git .`). It must report no leaks.

## 2. Build

Double-click **Build Both Versions.command** (or run `./build.sh --install`, `./"Make Beautify DMG.command"`,
`./build.sh --arch x86_64`, `./"Make Beautify DMG.command" intel`). It writes:

- `dist/ProtoArt-Beautify-<version>.dmg` (Apple Silicon) and `.dmg.sha256`
- `dist-intel/ProtoArt-Beautify-<version>-Intel.dmg` (Intel) and `.dmg.sha256`

The build and DMG scripts warn if the bundle still mentions your home folder (it must say "none found"). Open each DMG once and check the app starts.

## 3. Publish (only after review)

```bash
git tag -a v<version> -m "ProtoArt Beautify <version>"
git push origin main --tags
gh release create v<version> dist/*.dmg dist/*.dmg.sha256 dist-intel/*.dmg dist-intel/*.dmg.sha256 \
  --title "ProtoArt Beautify <version>" --notes-file <notes.md>
```

Release notes should say: macOS 13+; Apple Silicon DMG vs Intel DMG; not notarised (Open Anyway once); no model
weights included and each model has its own licence (4x-UltraSharp is non-commercial); the SHA-256 of each DMG.
