# 🚀 VeZ Release Setup Guide

Complete guide for setting up automated releases with GitHub Actions.

---

## Overview

This guide will help you set up:
1. ✅ Automated cross-platform binary builds
2. ✅ GitHub Releases with downloadable binaries
3. ✅ One-liner installation scripts
4. ✅ Website download buttons with platform detection

---

## Prerequisites

- GitHub repository for VeZ
- GitHub account with repository access
- Repository secrets configured (if needed)

---

## Step 1: Update Repository References

Replace `YOUR_USERNAME` with your actual GitHub username in these files:

### Files to Update:

1. **`install.sh`** (Line 73)
   ```bash
   LATEST_VERSION=$(curl -fsSL https://api.github.com/repos/YOUR_USERNAME/VeZ/releases/latest | grep '"tag_name":' | sed -E 's/.*"([^"]+)".*/\1/')
   ```

2. **`install.ps1`** (Line 25)
   ```powershell
   $LatestRelease = Invoke-RestMethod -Uri "https://api.github.com/repos/YOUR_USERNAME/VeZ/releases/latest"
   ```

3. **`docs/index.html`** (Line 670)
   ```javascript
   const GITHUB_REPO = 'YOUR_USERNAME/VeZ';
   ```

4. **`INSTALLATION.md`** (Multiple locations)
   - Replace all instances of `YOUR_USERNAME` with your GitHub username

### Quick Find & Replace:

```bash
# Linux/macOS
find . -type f \( -name "install.sh" -o -name "install.ps1" -o -name "index.html" -o -name "INSTALLATION.md" \) -exec sed -i 's/YOUR_USERNAME/actual-username/g' {} +

# Or manually update each file
```

---

## Step 2: Configure GitHub Actions

The workflow file is already created at `.github/workflows/release.yml`.

### Workflow Features:
- ✅ Builds for Linux (x86_64, ARM64)
- ✅ Builds for macOS (Intel, Apple Silicon)
- ✅ Builds for Windows (x64)
- ✅ Creates GitHub Releases automatically
- ✅ Uploads binaries as release assets

### Trigger a Release:

```bash
# Create and push a version tag
git tag v1.0.0
git push origin v1.0.0

# Or create a release through GitHub UI
# The workflow will automatically build and publish binaries
```

---

## Step 3: Verify Workflow Permissions

Ensure your GitHub Actions have the correct permissions:

1. Go to: `Settings` → `Actions` → `General`
2. Under "Workflow permissions", select:
   - ✅ **Read and write permissions**
3. Save changes

---

## Step 4: Test the Release Process

### Manual Workflow Trigger:

1. Go to: `Actions` → `Build and Release VeZ`
2. Click: `Run workflow`
3. Select branch: `main`
4. Click: `Run workflow`

This will test the build process without creating a release.

### Create a Test Release:

```bash
# Create a test tag
git tag v0.1.0-test
git push origin v0.1.0-test
```

Monitor the workflow at: `https://github.com/YOUR_USERNAME/VeZ/actions`

---

## Step 5: Verify Release Assets

After a successful release, verify the following files are uploaded:

### Expected Release Assets:

```
vez-v1.0.0-x86_64-unknown-linux-gnu.tar.gz
vez-v1.0.0-aarch64-unknown-linux-gnu.tar.gz
vez-v1.0.0-x86_64-apple-darwin.zip
vez-v1.0.0-aarch64-apple-darwin.zip
vez-v1.0.0-x86_64-pc-windows-msvc.zip
```

Each archive should contain:
- `vezc` (or `vezc.exe` on Windows)
- `vpm` (or `vpm.exe`)
- `vez-lsp` (or `vez-lsp.exe`)
- `README.md`
- `LICENSE`
- `QUICK_START.md`
- `INSTALL.md`
- `AI_TOOLING.md`

---

## Step 6: Test Installation Scripts

### Test Linux/macOS Installer:

```bash
# Test locally first
bash install.sh

# Test from GitHub (after release)
curl -fsSL https://raw.githubusercontent.com/YOUR_USERNAME/VeZ/main/install.sh | bash
```

### Test Windows Installer:

```powershell
# Test locally first
.\install.ps1

# Test from GitHub (after release)
iwr -useb https://raw.githubusercontent.com/YOUR_USERNAME/VeZ/main/install.ps1 | iex
```

---

## Step 7: Update Website

The website (`docs/index.html`) is already configured with:
- ✅ Platform detection
- ✅ Download buttons for all platforms
- ✅ One-liner installation commands
- ✅ Copy-to-clipboard functionality

### Deploy Website:

If using GitHub Pages:

1. Go to: `Settings` → `Pages`
2. Source: `Deploy from a branch`
3. Branch: `main` → `/docs`
4. Save

Your site will be available at: `https://YOUR_USERNAME.github.io/VeZ/`

---

## Step 8: Create Release Notes Template

Create `.github/RELEASE_TEMPLATE.md`:

```markdown
## 🎉 VeZ vX.Y.Z Released!

### ✨ New Features
- Feature 1
- Feature 2

### 🐛 Bug Fixes
- Fix 1
- Fix 2

### 📦 Installation

**Linux & macOS:**
```bash
curl -fsSL https://raw.githubusercontent.com/YOUR_USERNAME/VeZ/main/install.sh | bash
```

**Windows:**
```powershell
iwr -useb https://raw.githubusercontent.com/YOUR_USERNAME/VeZ/main/install.ps1 | iex
```

### 📥 Direct Downloads

Choose your platform:
- [Linux x64](link)
- [Linux ARM64](link)
- [macOS Intel](link)
- [macOS Apple Silicon](link)
- [Windows x64](link)

### 📚 Documentation
- [Installation Guide](INSTALLATION.md)
- [Quick Start](QUICK_START.md)
- [API Documentation](docs/)

### 🙏 Contributors
Thanks to all contributors!
```

---

## Step 9: Automate Release Notes

Add to `.github/workflows/release.yml`:

```yaml
- name: Generate Release Notes
  run: |
    echo "## What's Changed" > release_notes.md
    git log $(git describe --tags --abbrev=0 HEAD^)..HEAD --pretty=format:"- %s" >> release_notes.md
```

Then update the release step to use the notes:

```yaml
- name: Create GitHub Release
  uses: softprops/action-gh-release@v1
  with:
    body_path: release_notes.md
    files: artifacts/**/*
```

---

## Step 10: Set Up Download Analytics (Optional)

Track downloads using GitHub API or third-party services:

### Using GitHub API:

```bash
# Get download count for a release
curl https://api.github.com/repos/YOUR_USERNAME/VeZ/releases/latest | \
  jq '.assets[] | {name: .name, downloads: .download_count}'
```

### Add to Website:

```javascript
// In docs/index.html
async function fetchDownloadStats() {
  const response = await fetch('https://api.github.com/repos/YOUR_USERNAME/VeZ/releases/latest');
  const data = await response.json();
  const totalDownloads = data.assets.reduce((sum, asset) => sum + asset.download_count, 0);
  document.getElementById('download-count').textContent = totalDownloads.toLocaleString();
}
```

---

## Maintenance

### Regular Tasks:

1. **Update Dependencies:**
   ```bash
   cargo update
   cargo audit
   ```

2. **Test Builds Locally:**
   ```bash
   cargo build --release --target x86_64-unknown-linux-gnu
   cargo build --release --target x86_64-apple-darwin
   cargo build --release --target x86_64-pc-windows-msvc
   ```

3. **Version Bumping:**
   ```bash
   # Update version in Cargo.toml
   # Then create tag
   git tag v1.1.0
   git push origin v1.1.0
   ```

---

## Troubleshooting

### Build Failures

**Check workflow logs:**
```
Actions → Build and Release VeZ → Select failed run → View logs
```

**Common issues:**
- Missing dependencies
- Compilation errors
- Permission issues

### Release Not Created

**Verify:**
1. Tag pushed to GitHub: `git ls-remote --tags origin`
2. Workflow permissions enabled
3. No errors in workflow logs

### Download Links Broken

**Check:**
1. Release exists: `https://github.com/YOUR_USERNAME/VeZ/releases`
2. Assets uploaded correctly
3. File names match expected pattern

---

## Security Best Practices

1. **Sign Releases:**
   ```bash
   # Generate GPG key
   gpg --gen-key
   
   # Sign tag
   git tag -s v1.0.0 -m "Release v1.0.0"
   ```

2. **Verify Checksums:**
   Add to workflow:
   ```yaml
   - name: Generate checksums
     run: |
       cd artifacts
       sha256sum **/* > SHA256SUMS
   ```

3. **Use Dependabot:**
   Create `.github/dependabot.yml`:
   ```yaml
   version: 2
   updates:
     - package-ecosystem: "cargo"
       directory: "/"
       schedule:
         interval: "weekly"
   ```

---

## Advanced Configuration

### Multi-Architecture Builds

Add more targets to `.github/workflows/release.yml`:

```yaml
matrix:
  include:
    - os: ubuntu-latest
      target: x86_64-unknown-linux-musl  # Static binary
    - os: ubuntu-latest
      target: armv7-unknown-linux-gnueabihf  # ARM32
```

### Custom Build Scripts

Create `scripts/build-release.sh`:

```bash
#!/bin/bash
set -e

TARGET=$1
cargo build --release --target $TARGET

# Strip binaries
strip target/$TARGET/release/vezc

# Create archive
tar -czf vez-$TARGET.tar.gz -C target/$TARGET/release vezc vpm vez-lsp
```

---

## Checklist

Before your first release:

- [ ] Updated all `YOUR_USERNAME` references
- [ ] Tested workflow locally
- [ ] Verified GitHub Actions permissions
- [ ] Created test release
- [ ] Tested installation scripts on all platforms
- [ ] Updated website with correct links
- [ ] Set up GitHub Pages (if using)
- [ ] Created release notes template
- [ ] Documented release process
- [ ] Set up monitoring/analytics

---

## Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Rust Cross-Compilation](https://rust-lang.github.io/rustup/cross-compilation.html)
- [GitHub Releases API](https://docs.github.com/en/rest/releases)
- [Semantic Versioning](https://semver.org/)

---

**Your automated release pipeline is ready!** 🎉

Every time you push a version tag, GitHub Actions will:
1. Build binaries for all platforms
2. Create a GitHub Release
3. Upload all binaries
4. Make them available for download

Users can then install VeZ with a single command! 🚀
