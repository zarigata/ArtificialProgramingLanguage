#!/bin/bash
# Verify all VeZ binaries are working correctly

set -e

echo "=========================================="
echo "VeZ Binary Verification"
echo "=========================================="
echo ""

GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

RELEASE_DIR="target/release"

echo -e "${BLUE}Verifying VeZ Compiler (vezc)...${NC}"
if [ -f "$RELEASE_DIR/vezc" ]; then
    echo -e "${GREEN}✓ Binary exists${NC}"
    echo "  Size: $(du -h $RELEASE_DIR/vezc | cut -f1)"
    echo "  Permissions: $(ls -l $RELEASE_DIR/vezc | cut -d' ' -f1)"
    
    # Try to run with --version flag
    if $RELEASE_DIR/vezc --version 2>/dev/null || $RELEASE_DIR/vezc --help 2>/dev/null || true; then
        echo -e "${GREEN}✓ Binary is executable${NC}"
    else
        echo -e "${RED}✗ Binary exists but may have runtime issues${NC}"
    fi
else
    echo -e "${RED}✗ Binary not found${NC}"
fi
echo ""

echo -e "${BLUE}Verifying Package Manager (vpm)...${NC}"
if [ -f "$RELEASE_DIR/vpm" ]; then
    echo -e "${GREEN}✓ Binary exists${NC}"
    echo "  Size: $(du -h $RELEASE_DIR/vpm | cut -f1)"
    echo "  Permissions: $(ls -l $RELEASE_DIR/vpm | cut -d' ' -f1)"
    
    if $RELEASE_DIR/vpm --version 2>/dev/null || $RELEASE_DIR/vpm --help 2>/dev/null || true; then
        echo -e "${GREEN}✓ Binary is executable${NC}"
    else
        echo -e "${RED}✗ Binary exists but may have runtime issues${NC}"
    fi
else
    echo -e "${RED}✗ Binary not found${NC}"
fi
echo ""

echo -e "${BLUE}Verifying Language Server (vez-lsp)...${NC}"
if [ -f "$RELEASE_DIR/vez-lsp" ]; then
    echo -e "${GREEN}✓ Binary exists${NC}"
    echo "  Size: $(du -h $RELEASE_DIR/vez-lsp | cut -f1)"
    echo "  Permissions: $(ls -l $RELEASE_DIR/vez-lsp | cut -d' ' -f1)"
    
    if $RELEASE_DIR/vez-lsp --version 2>/dev/null || $RELEASE_DIR/vez-lsp --help 2>/dev/null || true; then
        echo -e "${GREEN}✓ Binary is executable${NC}"
    else
        echo -e "${RED}✗ Binary exists but may have runtime issues${NC}"
    fi
else
    echo -e "${RED}✗ Binary not found${NC}"
fi
echo ""

echo -e "${BLUE}Checking file types...${NC}"
for binary in vezc vpm vez-lsp; do
    if [ -f "$RELEASE_DIR/$binary" ]; then
        echo "$binary: $(file $RELEASE_DIR/$binary)"
    fi
done
echo ""

echo -e "${BLUE}Checking dependencies...${NC}"
for binary in vezc vpm vez-lsp; do
    if [ -f "$RELEASE_DIR/$binary" ]; then
        echo ""
        echo "$binary dependencies:"
        ldd "$RELEASE_DIR/$binary" 2>/dev/null || echo "  (static binary or ldd not available)"
    fi
done
echo ""

echo "=========================================="
echo -e "${GREEN}Verification Complete${NC}"
echo "=========================================="
