#!/bin/bash
# Simple build script to see actual compilation status

echo "=========================================="
echo "VeZ Simple Build Test"
echo "=========================================="
echo ""

# Build just the compiler first
echo "Building compiler..."
cargo build --package vez_compiler 2>&1

echo ""
echo "Build complete. Checking binaries..."
echo ""

# Check what was built
if [ -f "target/debug/vezc" ]; then
    echo "✓ vezc binary created:"
    ls -lh target/debug/vezc
    file target/debug/vezc
else
    echo "✗ vezc binary NOT created"
fi

echo ""
echo "Checking library..."
if [ -f "target/debug/libvez_compiler.rlib" ]; then
    echo "✓ Compiler library created:"
    ls -lh target/debug/libvez_compiler.rlib
else
    echo "✗ Compiler library NOT created"
fi

echo ""
echo "Running a simple test..."
cargo test --package vez_compiler --lib lexer::tests::test_tokenize_number -- --nocapture 2>&1 | tail -20

echo ""
echo "=========================================="
