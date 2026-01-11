# VeZ Installation Guide

This guide walks you through installing the VeZ toolchain from GitHub Releases or from source, across Linux, macOS (Intel/Apple Silicon), and Windows. It also covers minimal requirements for AI tools to consume the artifacts.

## Quick Install (Prebuilt Binaries)

1. **Download** the archive for your platform from GitHub Releases:
   - `vez-<version>-x86_64-unknown-linux-gnu.tar.gz`
   - `vez-<version>-x86_64-apple-darwin.zip`
   - `vez-<version>-aarch64-apple-darwin.zip`
   - `vez-<version>-x86_64-pc-windows-msvc.zip`

2. **Unpack** into a directory on your PATH:
   ```bash
   # Linux / macOS
   tar -xzf vez-<version>-<target>.tar.gz -C ~/.local/opt/vez
   echo 'export PATH="$HOME/.local/opt/vez:<REPLACE_WITH_EXACT_DIR>:$PATH"' >> ~/.bashrc
   source ~/.bashrc
   ```
   ```powershell
   # Windows (PowerShell)
   Expand-Archive vez-<version>-x86_64-pc-windows-msvc.zip -DestinationPath "$Env:LOCALAPPDATA\vez"
   setx PATH "$Env:LOCALAPPDATA\vez;$Env:PATH"
   ```

3. **Verify**:
   ```bash
   vezc --version
   vpm --help        # optional tool
   vez-lsp --help    # optional tool
   ```

## Install From Source

Requirements:
- Rust toolchain (stable)
- Git

Steps:
```bash
git clone https://github.com/<your-org>/vez.git
cd vez
cargo build --workspace --release
```

Artifacts will be in `target/release/`:
- `vezc` (compiler)
- `vpm` (package manager)
- `vez-lsp` (language server)

Add the directory to PATH:
```bash
echo 'export PATH="$PWD/target/release:$PATH"' >> ~/.bashrc
```

## Platform Notes

- **Linux**: Prebuilt with `x86_64-unknown-linux-gnu`. Ensure `glibc` ≥ 2.31 for best compatibility.
- **macOS**: Universal support via separate `x86_64` and `aarch64` artifacts.
- **Windows**: MSVC toolchain. Use PowerShell for extracting and PATH updates.

## Checksums

Each release attaches archives only. Recommended:
```bash
shasum -a 256 vez-<version>-<target>.tar.gz
```
Compare with the published checksum list if provided in the release notes.

## Uninstall

Remove the install directory and PATH entry:
```bash
rm -rf ~/.local/opt/vez
```
Adjust shell init file to remove the PATH addition.

## Troubleshooting

- If `vezc` is not found, confirm PATH is updated and restart your shell.
- If binaries fail to run, ensure you downloaded the correct target triple for your OS/CPU.
- For source builds, ensure Rust toolchain is installed and up to date: `rustup update`.
