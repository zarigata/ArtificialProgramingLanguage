#!/bin/bash
# Quick test to verify VeZ components compile and work

set -e

echo "=========================================="
echo "VeZ Quick Build & Test"
echo "=========================================="
echo ""

# Try to build just the compiler library first
echo "Building compiler library..."
cargo build --package vez_compiler --lib 2>&1 | tail -20

echo ""
echo "Build status: $?"
echo ""

# Check if binary exists
echo "Checking for binaries..."
ls -lh target/debug/vezc 2>/dev/null || echo "vezc not found yet"
ls -lh target/debug/vpm 2>/dev/null || echo "vpm not found yet"
ls -lh target/debug/vez-lsp 2>/dev/null || echo "vez-lsp not found yet"

echo ""
echo "Running quick tests..."
cargo test --package vez_compiler --lib -- --test-threads=1 2>&1 | tail -30

echo ""
echo "=========================================="
echo "Quick test complete!"
echo "=========================================="
