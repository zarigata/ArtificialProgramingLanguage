# 🚀 VeZ Installation Guide

Complete installation instructions for all platforms.

---

## Quick Install (Recommended)

### Linux & macOS

```bash
curl -fsSL https://raw.githubusercontent.com/YOUR_USERNAME/VeZ/main/install.sh | bash
```

### Windows (PowerShell)

```powershell
iwr -useb https://raw.githubusercontent.com/YOUR_USERNAME/VeZ/main/install.ps1 | iex
```

---

## Platform-Specific Instructions

### 🐧 Linux

#### One-Line Install
```bash
curl -fsSL https://raw.githubusercontent.com/YOUR_USERNAME/VeZ/main/install.sh | bash
```

#### Manual Installation

1. **Download the latest release:**
   ```bash
   # For x86_64
   wget https://github.com/YOUR_USERNAME/VeZ/releases/latest/download/vez-VERSION-x86_64-unknown-linux-gnu.tar.gz
   
   # For ARM64
   wget https://github.com/YOUR_USERNAME/VeZ/releases/latest/download/vez-VERSION-aarch64-unknown-linux-gnu.tar.gz
   ```

2. **Extract the archive:**
   ```bash
   tar -xzf vez-*.tar.gz
   cd vez-*/x86_64-unknown-linux-gnu  # or aarch64-unknown-linux-gnu
   ```

3. **Install binaries:**
   ```bash
   sudo cp vezc vpm vez-lsp /usr/local/bin/
   sudo chmod +x /usr/local/bin/{vezc,vpm,vez-lsp}
   ```

4. **Verify installation:**
   ```bash
   vezc --version
   vpm --version
   ```

#### Supported Distributions
- ✅ Ubuntu 20.04+
- ✅ Debian 11+
- ✅ Fedora 35+
- ✅ Arch Linux
- ✅ openSUSE
- ✅ Any Linux with glibc 2.31+

---

### 🍎 macOS

#### One-Line Install
```bash
curl -fsSL https://raw.githubusercontent.com/YOUR_USERNAME/VeZ/main/install.sh | bash
```

#### Homebrew (Coming Soon)
```bash
brew install vez
```

#### Manual Installation

1. **Download the latest release:**
   ```bash
   # For Intel Macs
   curl -LO https://github.com/YOUR_USERNAME/VeZ/releases/latest/download/vez-VERSION-x86_64-apple-darwin.zip
   
   # For Apple Silicon (M1/M2/M3)
   curl -LO https://github.com/YOUR_USERNAME/VeZ/releases/latest/download/vez-VERSION-aarch64-apple-darwin.zip
   ```

2. **Extract the archive:**
   ```bash
   unzip vez-*.zip
   cd vez-*/x86_64-apple-darwin  # or aarch64-apple-darwin
   ```

3. **Install binaries:**
   ```bash
   sudo cp vezc vpm vez-lsp /usr/local/bin/
   sudo chmod +x /usr/local/bin/{vezc,vpm,vez-lsp}
   ```

4. **Verify installation:**
   ```bash
   vezc --version
   vpm --version
   ```

#### Supported Versions
- ✅ macOS 11 (Big Sur) and later
- ✅ Intel and Apple Silicon

---

### 🪟 Windows

#### One-Line Install (PowerShell)
```powershell
iwr -useb https://raw.githubusercontent.com/YOUR_USERNAME/VeZ/main/install.ps1 | iex
```

#### Manual Installation

1. **Download the latest release:**
   - Visit: https://github.com/YOUR_USERNAME/VeZ/releases/latest
   - Download: `vez-VERSION-x86_64-pc-windows-msvc.zip`

2. **Extract the archive:**
   - Right-click the ZIP file → Extract All
   - Or use PowerShell:
     ```powershell
     Expand-Archive -Path vez-*.zip -DestinationPath C:\VeZ
     ```

3. **Add to PATH:**
   - Open System Properties → Environment Variables
   - Edit the `Path` variable for your user
   - Add: `C:\VeZ\x86_64-pc-windows-msvc`
   - Or use PowerShell:
     ```powershell
     $env:Path += ";C:\VeZ\x86_64-pc-windows-msvc"
     [Environment]::SetEnvironmentVariable("Path", $env:Path, "User")
     ```

4. **Restart your terminal** and verify:
   ```powershell
   vezc --version
   vpm --version
   ```

#### Supported Versions
- ✅ Windows 10 (1809+)
- ✅ Windows 11
- ✅ Windows Server 2019+

---

## What Gets Installed

The VeZ installation includes three main tools:

### 1. **vezc** - VeZ Compiler
The main compiler that transforms VeZ source code into optimized binaries.

```bash
vezc --help
vezc program.vez          # Compile a VeZ program
vezc --style python app.pyvez  # Compile Python-style VeZ
vezc --release program.vez     # Optimized release build
```

### 2. **vpm** - VeZ Package Manager
Manage dependencies, create projects, and publish packages.

```bash
vpm --help
vpm new my-project        # Create new project
vpm add serde            # Add dependency
vpm build                # Build project
vpm test                 # Run tests
vpm publish              # Publish package
```

### 3. **vez-lsp** - Language Server Protocol
Powers IDE features like autocomplete, go-to-definition, and error checking.

```bash
vez-lsp --help
# Usually configured automatically by your IDE/editor
```

---

## Verify Installation

After installation, verify everything works:

```bash
# Check versions
vezc --version
vpm --version
vez-lsp --version

# Create a test program
echo 'fn main() { println("Hello, VeZ!"); }' > hello.vez

# Compile and run
vezc hello.vez
./hello  # Linux/macOS
# or
.\hello.exe  # Windows
```

Expected output:
```
Hello, VeZ!
```

---

## IDE/Editor Setup

### Visual Studio Code

1. Install the VeZ extension (coming soon):
   ```bash
   code --install-extension vez-lang.vez-vscode
   ```

2. Or configure manually in `settings.json`:
   ```json
   {
     "vez.languageServer.path": "/usr/local/bin/vez-lsp"
   }
   ```

### Vim/Neovim

Add to your config:
```vim
" Using vim-plug
Plug 'vez-lang/vez.vim'

" LSP configuration (with coc.nvim)
let g:coc_global_extensions = ['coc-vez']
```

### Emacs

```elisp
;; Add to your init.el
(use-package vez-mode
  :ensure t
  :mode "\\.vez\\'")

(use-package lsp-mode
  :hook (vez-mode . lsp))
```

---

## Updating VeZ

### Automatic Update

```bash
# Linux/macOS
curl -fsSL https://raw.githubusercontent.com/YOUR_USERNAME/VeZ/main/install.sh | bash

# Windows
iwr -useb https://raw.githubusercontent.com/YOUR_USERNAME/VeZ/main/install.ps1 | iex
```

### Using VPM

```bash
vpm self-update
```

---

## Uninstallation

### Linux/macOS

```bash
sudo rm /usr/local/bin/{vezc,vpm,vez-lsp}
```

### Windows

```powershell
# Remove installation directory
Remove-Item -Recurse -Force "$env:LOCALAPPDATA\VeZ"

# Remove from PATH (manual step required)
# Edit Environment Variables and remove VeZ path
```

---

## Troubleshooting

### Command not found

**Linux/macOS:**
- Ensure `/usr/local/bin` is in your PATH:
  ```bash
  echo $PATH | grep /usr/local/bin
  ```
- If not, add to `~/.bashrc` or `~/.zshrc`:
  ```bash
  export PATH="/usr/local/bin:$PATH"
  ```

**Windows:**
- Restart your terminal after installation
- Verify PATH includes VeZ directory:
  ```powershell
  $env:Path -split ';' | Select-String VeZ
  ```

### Permission denied

**Linux/macOS:**
- Use `sudo` for installation to `/usr/local/bin`
- Or install to user directory:
  ```bash
  mkdir -p ~/.local/bin
  cp vezc vpm vez-lsp ~/.local/bin/
  export PATH="$HOME/.local/bin:$PATH"
  ```

### SSL/TLS errors

```bash
# Update certificates (Ubuntu/Debian)
sudo apt-get update && sudo apt-get install ca-certificates

# macOS
brew install ca-certificates

# Windows
# Download from official release page instead
```

### Architecture mismatch

Ensure you download the correct binary for your system:
- Linux x64: `x86_64-unknown-linux-gnu`
- Linux ARM64: `aarch64-unknown-linux-gnu`
- macOS Intel: `x86_64-apple-darwin`
- macOS Apple Silicon: `aarch64-apple-darwin`
- Windows: `x86_64-pc-windows-msvc`

---

## Building from Source

If you prefer to build from source:

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/VeZ.git
cd VeZ

# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Build release binaries
cargo build --release --workspace

# Install
sudo cp target/release/{vezc,vpm,vez-lsp} /usr/local/bin/
```

---

## System Requirements

### Minimum Requirements
- **OS:** Linux (glibc 2.31+), macOS 11+, Windows 10 (1809+)
- **RAM:** 2 GB
- **Disk:** 100 MB for binaries
- **Architecture:** x86_64 or ARM64

### Recommended
- **RAM:** 4 GB or more
- **Disk:** 500 MB for full toolchain + packages
- **CPU:** Multi-core for parallel compilation

---

## Getting Help

- **Documentation:** https://github.com/YOUR_USERNAME/VeZ/tree/main/docs
- **Issues:** https://github.com/YOUR_USERNAME/VeZ/issues
- **Discord:** https://discord.gg/vez-lang (coming soon)
- **Email:** support@vez-lang.org

---

## Next Steps

After installation:

1. **Read the Quick Start Guide:**
   ```bash
   cat /usr/local/share/vez/QUICK_START.md
   ```

2. **Try the examples:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/VeZ.git
   cd VeZ/examples
   vezc fibonacci.vez
   ```

3. **Create your first project:**
   ```bash
   vpm new my-first-vez-project
   cd my-first-vez-project
   vpm build
   ```

4. **Explore the documentation:**
   - Language specification
   - Standard library
   - Best practices
   - Advanced features

---

**Happy coding with VeZ!** 🚀
