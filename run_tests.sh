#!/bin/bash
# VeZ Comprehensive Test Suite

set -e

echo "=========================================="
echo "VeZ Programming Language - Test Suite"
echo "=========================================="
echo ""

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Test counters
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

echo -e "${BLUE}Running comprehensive test suite...${NC}"
echo ""

echo -e "${BLUE}Test 1: Compiler Library Tests${NC}"
cargo test --package vez_compiler --lib 2>&1 | tee -a test_output.log
echo ""

echo -e "${BLUE}Test 2: Lexer Tests${NC}"
cargo test --package vez_compiler --lib lexer 2>&1 | tee -a test_output.log
echo ""

echo -e "${BLUE}Test 3: Parser Tests${NC}"
cargo test --package vez_compiler --lib parser 2>&1 | tee -a test_output.log
echo ""

echo -e "${BLUE}Test 4: Semantic Analysis Tests${NC}"
cargo test --package vez_compiler --lib semantic 2>&1 | tee -a test_output.log
echo ""

echo -e "${BLUE}Test 5: Borrow Checker Tests${NC}"
cargo test --package vez_compiler --lib borrow 2>&1 | tee -a test_output.log
echo ""

echo -e "${BLUE}Test 6: IR Generation Tests${NC}"
cargo test --package vez_compiler --lib ir 2>&1 | tee -a test_output.log
echo ""

echo -e "${BLUE}Test 7: Optimizer Tests${NC}"
cargo test --package vez_compiler --lib optimizer 2>&1 | tee -a test_output.log
echo ""

echo -e "${BLUE}Test 8: LLVM Backend Tests${NC}"
cargo test --package vez_compiler --lib codegen 2>&1 | tee -a test_output.log
echo ""

echo -e "${BLUE}Test 9: Macro System Tests${NC}"
cargo test --package vez_compiler --lib macro_system 2>&1 | tee -a test_output.log
echo ""

echo -e "${BLUE}Test 10: Async Runtime Tests${NC}"
cargo test --package vez_compiler --lib async_runtime 2>&1 | tee -a test_output.log
echo ""

echo -e "${BLUE}Test 11: Verification System Tests${NC}"
cargo test --package vez_compiler --lib verification 2>&1 | tee -a test_output.log
echo ""

echo -e "${BLUE}Test 12: GPU Backend Tests${NC}"
cargo test --package vez_compiler --lib gpu 2>&1 | tee -a test_output.log
echo ""

echo -e "${BLUE}Test 13: Compile-Time Evaluation Tests${NC}"
cargo test --package vez_compiler --lib consteval 2>&1 | tee -a test_output.log
echo ""

echo -e "${BLUE}Test 14: Plugin System Tests${NC}"
cargo test --package vez_compiler --lib plugin 2>&1 | tee -a test_output.log
echo ""

echo -e "${BLUE}Test 15: All Workspace Tests${NC}"
cargo test --workspace --all-features 2>&1 | tee -a test_output.log
echo ""

echo -e "${BLUE}Test 16: Documentation Tests${NC}"
cargo test --doc --workspace 2>&1 | tee -a test_output.log
echo ""

echo "=========================================="
echo -e "${GREEN}Test Summary${NC}"
echo "=========================================="
echo ""

# Extract test statistics from output
if [ -f test_output.log ]; then
    echo "Test output saved to: test_output.log"
    echo ""
    
    # Count test results
    PASSED=$(grep -c "test result: ok" test_output.log || echo "0")
    FAILED=$(grep -c "test result: FAILED" test_output.log || echo "0")
    
    echo -e "${GREEN}Passed test suites: $PASSED${NC}"
    echo -e "${RED}Failed test suites: $FAILED${NC}"
    echo ""
    
    # Show any failures
    if [ $FAILED -gt 0 ]; then
        echo -e "${RED}Failed tests:${NC}"
        grep "FAILED" test_output.log || echo "No specific failures found"
        echo ""
    fi
fi

echo -e "${GREEN}Test suite complete!${NC}"
echo "=========================================="
