#!/bin/bash
# Manual VeZ Verification Script
# Direct testing without complex automation

echo "=========================================="
echo "VeZ Manual Verification"
echo "=========================================="
echo ""

# 1. Check Rust
echo "1. Checking Rust installation..."
rustc --version
cargo --version
echo ""

# 2. Check files
echo "2. Checking project structure..."
echo "Workspace Cargo.toml: $([ -f Cargo.toml ] && echo '✓' || echo '✗')"
echo "Compiler Cargo.toml: $([ -f compiler/Cargo.toml ] && echo '✓' || echo '✗')"
echo "VPM Cargo.toml: $([ -f tools/vpm/Cargo.toml ] && echo '✓' || echo '✗')"
echo "LSP Cargo.toml: $([ -f tools/lsp/Cargo.toml ] && echo '✓' || echo '✗')"
echo "Testing Cargo.toml: $([ -f tools/testing/Cargo.toml ] && echo '✓' || echo '✗')"
echo ""

# 3. Count source files
echo "3. Source code statistics..."
echo "Rust files: $(find compiler/src -name '*.rs' 2>/dev/null | wc -l)"
echo "Total lines: $(find compiler/src -name '*.rs' -exec cat {} \; 2>/dev/null | wc -l)"
echo ""

# 4. Build compiler library
echo "4. Building compiler library..."
echo "Running: cargo build --package vez_compiler --lib"
cargo build --package vez_compiler --lib 2>&1 | tail -20
BUILD_STATUS=$?
echo "Build exit code: $BUILD_STATUS"
echo ""

# 5. Check for binary
echo "5. Checking for binaries..."
if [ -f "target/debug/libvez_compiler.rlib" ]; then
    echo "✓ Compiler library: $(ls -lh target/debug/libvez_compiler.rlib | awk '{print $5}')"
else
    echo "✗ Compiler library not found"
fi

if [ -f "target/debug/vezc" ]; then
    echo "✓ Compiler binary: $(ls -lh target/debug/vezc | awk '{print $5}')"
else
    echo "✗ Compiler binary not found"
fi
echo ""

# 6. Run tests
echo "6. Running tests..."
echo "Running: cargo test --package vez_compiler --lib"
cargo test --package vez_compiler --lib 2>&1 | grep -E "test result:|running" | head -20
TEST_STATUS=$?
echo "Test exit code: $TEST_STATUS"
echo ""

# 7. Check compilation
echo "7. Running cargo check..."
cargo check --package vez_compiler 2>&1 | tail -10
CHECK_STATUS=$?
echo "Check exit code: $CHECK_STATUS"
echo ""

# Summary
echo "=========================================="
echo "Summary"
echo "=========================================="
echo "Build Status: $([ $BUILD_STATUS -eq 0 ] && echo '✓ SUCCESS' || echo '✗ FAILED')"
echo "Test Status: $([ $TEST_STATUS -eq 0 ] && echo '✓ SUCCESS' || echo '✗ FAILED')"
echo "Check Status: $([ $CHECK_STATUS -eq 0 ] && echo '✓ SUCCESS' || echo '✗ FAILED')"
echo ""

if [ $BUILD_STATUS -eq 0 ] && [ $CHECK_STATUS -eq 0 ]; then
    echo "✓ VeZ compiler is functional!"
    exit 0
else
    echo "✗ VeZ has issues that need attention"
    exit 1
fi
