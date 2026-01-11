# VeZ Release Process (GitHub Actions)

This project ships tagged releases via GitHub Actions. When you push a tag like `v1.0.0`, CI builds artifacts for Linux, macOS (Intel + Apple Silicon), and Windows, then publishes a GitHub Release with binaries attached.

## Prerequisites
- Ensure `Cargo.toml` version matches the tag you plan to push.
- Commit all changes (especially docs/INSTALL.md, docs/AI_TOOLING.md, README.md).

## Steps to Cut a Release
```bash
# 1) Run tests locally (optional but recommended)
cargo build --workspace --release
cargo test --workspace

# 2) Tag the release (use semantic versioning)
git tag v1.0.0

# 3) Push the tag to trigger CI release
git push origin v1.0.0
```

## What CI Does
- Builds release binaries for:
  - `x86_64-unknown-linux-gnu`
  - `x86_64-apple-darwin`
  - `aarch64-apple-darwin`
  - `x86_64-pc-windows-msvc`
- Packages artifacts with README, LICENSE, QUICK_START, INSTALL, and AI_TOOLING guides.
- Publishes a GitHub Release with all archives attached.

## Artifacts Produced
- `vez-<version>-x86_64-unknown-linux-gnu.tar.gz`
- `vez-<version>-x86_64-apple-darwin.zip`
- `vez-<version>-aarch64-apple-darwin.zip`
- `vez-<version>-x86_64-pc-windows-msvc.zip`

## Optional: Checksums
If you want to publish checksums, after CI completes:
```bash
sha256sum vez-<version>-*.{tar.gz,zip} > checksums.txt
# Upload checksums.txt to the GitHub release manually (or extend the workflow).
```

## Verification
- Download the artifact for your platform and run:
  ```bash
  ./vezc --version
  ./vpm --help   # if present
  ./vez-lsp --help
  ```

## Notes
- Workflow file: `.github/workflows/release.yml`
- Release docs: `docs/INSTALL.md`, `docs/AI_TOOLING.md`
- If you need to re-run CI for the same tag, delete the GitHub release and the tag (locally and remotely), then re-push.
