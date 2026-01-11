#!/bin/bash
# VeZ Build Script - Build all binaries and run tests

set -e

echo "=========================================="
echo "VeZ Programming Language - Build System"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Build directory
BUILD_DIR="target"
RELEASE_DIR="$BUILD_DIR/release"
DEBUG_DIR="$BUILD_DIR/debug"

echo -e "${BLUE}Step 1: Checking Rust installation...${NC}"
if ! command -v cargo &> /dev/null; then
    echo -e "${RED}Error: Cargo not found. Please install Rust from https://rustup.rs/${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Rust/Cargo found: $(cargo --version)${NC}"
echo ""

echo -e "${BLUE}Step 2: Cleaning previous builds...${NC}"
cargo clean
echo -e "${GREEN}✓ Clean complete${NC}"
echo ""

echo -e "${BLUE}Step 3: Building compiler (vezc) in debug mode...${NC}"
cargo build --package vez_compiler --bin vezc
if [ -f "$DEBUG_DIR/vezc" ]; then
    echo -e "${GREEN}✓ Compiler binary built: $DEBUG_DIR/vezc${NC}"
    ls -lh "$DEBUG_DIR/vezc"
else
    echo -e "${RED}✗ Compiler binary not found${NC}"
fi
echo ""

echo -e "${BLUE}Step 4: Building package manager (vpm) in debug mode...${NC}"
cargo build --package vpm
if [ -f "$DEBUG_DIR/vpm" ]; then
    echo -e "${GREEN}✓ Package manager binary built: $DEBUG_DIR/vpm${NC}"
    ls -lh "$DEBUG_DIR/vpm"
else
    echo -e "${RED}✗ Package manager binary not found${NC}"
fi
echo ""

echo -e "${BLUE}Step 5: Building language server (vez-lsp) in debug mode...${NC}"
cargo build --package vez_lsp
if [ -f "$DEBUG_DIR/vez-lsp" ]; then
    echo -e "${GREEN}✓ Language server binary built: $DEBUG_DIR/vez-lsp${NC}"
    ls -lh "$DEBUG_DIR/vez-lsp"
else
    echo -e "${RED}✗ Language server binary not found${NC}"
fi
echo ""

echo -e "${BLUE}Step 6: Building testing framework in debug mode...${NC}"
cargo build --package vez_testing
if [ -f "$DEBUG_DIR/vez-test" ]; then
    echo -e "${GREEN}✓ Testing framework binary built: $DEBUG_DIR/vez-test${NC}"
    ls -lh "$DEBUG_DIR/vez-test"
else
    echo -e "${YELLOW}⚠ Testing framework binary not found (may be library only)${NC}"
fi
echo ""

echo -e "${BLUE}Step 7: Building all binaries in release mode (optimized)...${NC}"
cargo build --release --workspace
echo -e "${GREEN}✓ Release build complete${NC}"
echo ""

echo -e "${BLUE}Step 8: Listing all built binaries...${NC}"
echo ""
echo "Debug binaries:"
if [ -d "$DEBUG_DIR" ]; then
    ls -lh "$DEBUG_DIR"/{vezc,vpm,vez-lsp,vez-test} 2>/dev/null || echo "Some binaries not found"
fi
echo ""
echo "Release binaries:"
if [ -d "$RELEASE_DIR" ]; then
    ls -lh "$RELEASE_DIR"/{vezc,vpm,vez-lsp,vez-test} 2>/dev/null || echo "Some binaries not found"
fi
echo ""

echo -e "${BLUE}Step 9: Verifying binary executability...${NC}"
for binary in vezc vpm vez-lsp; do
    if [ -f "$RELEASE_DIR/$binary" ]; then
        if [ -x "$RELEASE_DIR/$binary" ]; then
            echo -e "${GREEN}✓ $binary is executable${NC}"
        else
            echo -e "${RED}✗ $binary is not executable${NC}"
        fi
    fi
done
echo ""

echo -e "${BLUE}Step 10: Running test suite...${NC}"
cargo test --workspace --no-fail-fast 2>&1 | tee test_results.log
TEST_EXIT_CODE=${PIPESTATUS[0]}
echo ""

if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
else
    echo -e "${YELLOW}⚠ Some tests failed (exit code: $TEST_EXIT_CODE)${NC}"
fi
echo ""

echo -e "${BLUE}Step 11: Running compiler tests specifically...${NC}"
cargo test --package vez_compiler 2>&1 | tee compiler_test_results.log
echo ""

echo -e "${BLUE}Step 12: Checking binary sizes...${NC}"
echo ""
echo "Binary sizes (release builds):"
if [ -d "$RELEASE_DIR" ]; then
    du -h "$RELEASE_DIR"/{vezc,vpm,vez-lsp} 2>/dev/null || echo "Some binaries not found"
fi
echo ""

echo "=========================================="
echo -e "${GREEN}Build Summary${NC}"
echo "=========================================="
echo ""
echo "Binaries built:"
echo "  - vezc (VeZ Compiler)"
echo "  - vpm (VeZ Package Manager)"
echo "  - vez-lsp (VeZ Language Server)"
echo ""
echo "Locations:"
echo "  Debug:   $DEBUG_DIR/"
echo "  Release: $RELEASE_DIR/"
echo ""
echo "Test results saved to:"
echo "  - test_results.log"
echo "  - compiler_test_results.log"
echo ""
echo -e "${GREEN}Build complete!${NC}"
echo "=========================================="
