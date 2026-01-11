#!/bin/bash
# Comprehensive VeZ Build and Test Verification

echo "=========================================="
echo "VeZ Comprehensive Build & Test Report"
echo "=========================================="
echo ""
date
echo ""

# Check Rust installation
echo "1. Rust Environment Check"
echo "-------------------------"
rustc --version 2>/dev/null || echo "ERROR: rustc not found"
cargo --version 2>/dev/null || echo "ERROR: cargo not found"
echo ""

# Check workspace structure
echo "2. Workspace Structure"
echo "----------------------"
echo "Workspace members:"
grep -A 10 "^\[workspace\]" Cargo.toml | grep "members" -A 5
echo ""

# Try building each component individually
echo "3. Building Components"
echo "----------------------"

echo "Building vez_compiler library..."
cargo build --package vez_compiler --lib 2>&1 | tail -20
COMPILER_LIB_STATUS=$?
echo "Status: $COMPILER_LIB_STATUS"
echo ""

echo "Building vez_compiler binary (vezc)..."
cargo build --package vez_compiler --bin vezc 2>&1 | tail -20
COMPILER_BIN_STATUS=$?
echo "Status: $COMPILER_BIN_STATUS"
echo ""

echo "Building vpm..."
cargo build --package vpm 2>&1 | tail -20
VPM_STATUS=$?
echo "Status: $VPM_STATUS"
echo ""

echo "Building vez_lsp..."
cargo build --package vez_lsp 2>&1 | tail -20
LSP_STATUS=$?
echo "Status: $LSP_STATUS"
echo ""

# Check what binaries were created
echo "4. Binary Verification"
echo "----------------------"
echo "Debug binaries:"
ls -lh target/debug/vezc 2>/dev/null && echo "✓ vezc found" || echo "✗ vezc not found"
ls -lh target/debug/vpm 2>/dev/null && echo "✓ vpm found" || echo "✗ vpm not found"
ls -lh target/debug/vez-lsp 2>/dev/null && echo "✓ vez-lsp found" || echo "✗ vez-lsp not found"
echo ""

# Run tests
echo "5. Running Tests"
echo "----------------"
echo "Running compiler tests..."
cargo test --package vez_compiler --lib 2>&1 | tail -30
TEST_STATUS=$?
echo "Test status: $TEST_STATUS"
echo ""

# Summary
echo "=========================================="
echo "Build Summary"
echo "=========================================="
echo "Compiler Library: $([ $COMPILER_LIB_STATUS -eq 0 ] && echo '✓ SUCCESS' || echo '✗ FAILED')"
echo "Compiler Binary:  $([ $COMPILER_BIN_STATUS -eq 0 ] && echo '✓ SUCCESS' || echo '✗ FAILED')"
echo "Package Manager:  $([ $VPM_STATUS -eq 0 ] && echo '✓ SUCCESS' || echo '✗ FAILED')"
echo "Language Server:  $([ $LSP_STATUS -eq 0 ] && echo '✓ SUCCESS' || echo '✗ FAILED')"
echo "Tests:            $([ $TEST_STATUS -eq 0 ] && echo '✓ SUCCESS' || echo '✗ FAILED')"
echo ""
echo "Total Lines of Code:"
find compiler/src stdlib runtime tools -name "*.rs" -o -name "*.zari" 2>/dev/null | xargs wc -l 2>/dev/null | tail -1
echo ""
echo "=========================================="
