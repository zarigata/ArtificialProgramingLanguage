#!/bin/bash
# VeZ One-Liner Installer for Linux and macOS
# Usage: curl -fsSL https://raw.githubusercontent.com/YOUR_USERNAME/VeZ/main/install.sh | bash

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}╔═══════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   VeZ Programming Language Installer  ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════╝${NC}"
echo ""

# Detect OS and architecture
OS="$(uname -s)"
ARCH="$(uname -m)"

case "$OS" in
    Linux*)
        OS_TYPE="linux"
        INSTALL_DIR="/usr/local/bin"
        ;;
    Darwin*)
        OS_TYPE="macos"
        INSTALL_DIR="/usr/local/bin"
        ;;
    *)
        echo -e "${RED}Error: Unsupported operating system: $OS${NC}"
        exit 1
        ;;
esac

case "$ARCH" in
    x86_64|amd64)
        ARCH_TYPE="x86_64"
        ;;
    aarch64|arm64)
        ARCH_TYPE="aarch64"
        ;;
    *)
        echo -e "${RED}Error: Unsupported architecture: $ARCH${NC}"
        exit 1
        ;;
esac

echo -e "${YELLOW}Detected OS: $OS_TYPE${NC}"
echo -e "${YELLOW}Detected Architecture: $ARCH_TYPE${NC}"
echo ""

# Determine target triple
if [ "$OS_TYPE" = "linux" ]; then
    TARGET="${ARCH_TYPE}-unknown-linux-gnu"
    ARCHIVE_EXT="tar.gz"
elif [ "$OS_TYPE" = "macos" ]; then
    TARGET="${ARCH_TYPE}-apple-darwin"
    ARCHIVE_EXT="zip"
fi

# Get latest release version
echo -e "${YELLOW}Fetching latest release...${NC}"
LATEST_VERSION=$(curl -fsSL https://api.github.com/repos/YOUR_USERNAME/VeZ/releases/latest | grep '"tag_name":' | sed -E 's/.*"([^"]+)".*/\1/')

if [ -z "$LATEST_VERSION" ]; then
    echo -e "${RED}Error: Could not determine latest version${NC}"
    exit 1
fi

echo -e "${GREEN}Latest version: $LATEST_VERSION${NC}"
echo ""

# Construct download URL
ARCHIVE_NAME="vez-${LATEST_VERSION}-${TARGET}.${ARCHIVE_EXT}"
DOWNLOAD_URL="https://github.com/YOUR_USERNAME/VeZ/releases/download/${LATEST_VERSION}/${ARCHIVE_NAME}"

echo -e "${YELLOW}Downloading VeZ...${NC}"
echo -e "${YELLOW}URL: $DOWNLOAD_URL${NC}"

# Create temporary directory
TMP_DIR=$(mktemp -d)
cd "$TMP_DIR"

# Download archive
if ! curl -fsSL -o "$ARCHIVE_NAME" "$DOWNLOAD_URL"; then
    echo -e "${RED}Error: Failed to download VeZ${NC}"
    rm -rf "$TMP_DIR"
    exit 1
fi

echo -e "${GREEN}Download complete!${NC}"
echo ""

# Extract archive
echo -e "${YELLOW}Extracting archive...${NC}"
if [ "$ARCHIVE_EXT" = "tar.gz" ]; then
    tar -xzf "$ARCHIVE_NAME"
else
    unzip -q "$ARCHIVE_NAME"
fi

# Install binaries
echo -e "${YELLOW}Installing VeZ to $INSTALL_DIR...${NC}"

# Check if we need sudo
if [ -w "$INSTALL_DIR" ]; then
    SUDO=""
else
    SUDO="sudo"
    echo -e "${YELLOW}Note: sudo access required for installation${NC}"
fi

# Install vezc (compiler)
if [ -f "$TARGET/vezc" ]; then
    $SUDO cp "$TARGET/vezc" "$INSTALL_DIR/vezc"
    $SUDO chmod +x "$INSTALL_DIR/vezc"
    echo -e "${GREEN}✓ Installed vezc (compiler)${NC}"
fi

# Install vpm (package manager)
if [ -f "$TARGET/vpm" ]; then
    $SUDO cp "$TARGET/vpm" "$INSTALL_DIR/vpm"
    $SUDO chmod +x "$INSTALL_DIR/vpm"
    echo -e "${GREEN}✓ Installed vpm (package manager)${NC}"
fi

# Install vez-lsp (language server)
if [ -f "$TARGET/vez-lsp" ]; then
    $SUDO cp "$TARGET/vez-lsp" "$INSTALL_DIR/vez-lsp"
    $SUDO chmod +x "$INSTALL_DIR/vez-lsp"
    echo -e "${GREEN}✓ Installed vez-lsp (language server)${NC}"
fi

# Cleanup
cd /
rm -rf "$TMP_DIR"

echo ""
echo -e "${GREEN}╔═══════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   VeZ Installation Complete! 🎉       ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}Installed binaries:${NC}"
echo -e "  • vezc     - VeZ compiler"
echo -e "  • vpm      - VeZ package manager"
echo -e "  • vez-lsp  - VeZ language server"
echo ""
echo -e "${YELLOW}Verify installation:${NC}"
echo -e "  vezc --version"
echo ""
echo -e "${YELLOW}Get started:${NC}"
echo -e "  vezc --help"
echo -e "  vpm --help"
echo ""
echo -e "${YELLOW}Documentation:${NC}"
echo -e "  https://github.com/YOUR_USERNAME/VeZ"
echo ""
echo -e "${GREEN}Happy coding with VeZ! 🚀${NC}"
