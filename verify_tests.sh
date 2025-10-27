#!/bin/bash
# Test Suite Verification Script

echo "🔍 Verifying Hermes Backend Test Suite Installation"
echo "=================================================="
echo ""

# Check Python
echo "✓ Checking Python version..."
python3 --version || python --version || { echo "❌ Python not found"; exit 1; }
echo ""

# Check pytest
echo "✓ Checking pytest installation..."
if ! command -v pytest &> /dev/null; then
    echo "❌ pytest not found. Run: pip install -e '.[test]'"
    exit 1
fi
pytest --version
echo ""

# Count test files
echo "✓ Counting test files..."
TEST_COUNT=$(find tests -name "test_*.py" | wc -l | tr -d ' ')
echo "   Found $TEST_COUNT test files"
echo ""

# Check test structure
echo "✓ Verifying test structure..."
declare -a required_dirs=(
    "tests/unit/hermes"
    "tests/unit/prism"
    "tests/integration"
    "tests/performance"
    "tests/fixtures"
    "tests/utils"
)

for dir in "${required_dirs[@]}"; do
    if [ -d "$dir" ]; then
        echo "   ✓ $dir"
    else
        echo "   ❌ $dir missing"
        exit 1
    fi
done
echo ""

# Check critical files
echo "✓ Checking critical files..."
declare -a required_files=(
    "pytest.ini"
    "tests/conftest.py"
    "tests/README.md"
    "tests/integration/test_vector_db_integration.py"
    ".github/workflows/test.yml"
)

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "   ✓ $file"
    else
        echo "   ❌ $file missing"
        exit 1
    fi
done
echo ""

# Try running a simple test
echo "✓ Running sample unit tests (dry-run)..."
if pytest tests/unit/hermes/test_models.py::TestUserIdentity::test_user_identity_creation --collect-only &> /dev/null; then
    echo "   ✓ Test discovery working"
else
    echo "   ❌ Test discovery failed"
    exit 1
fi
echo ""

# Check environment variables for integration tests
echo "✓ Checking integration test environment..."
if [ -z "$GOOGLE_API_KEY" ] || [ -z "$SUPABASE_URL" ] || [ -z "$SUPABASE_SERVICE_ROLE_KEY" ]; then
    echo "   ⚠️  Environment variables not set (integration tests will be skipped)"
    echo "      Set: GOOGLE_API_KEY, SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY"
else
    echo "   ✓ Environment variables configured"
fi
echo ""

# Summary
echo "=================================================="
echo "✅ Test Suite Verification Complete!"
echo ""
echo "📊 Summary:"
echo "   • $TEST_COUNT test files found"
echo "   • All required directories present"
echo "   • All critical files present"
echo "   • Test discovery working"
echo ""
echo "🚀 Next Steps:"
echo "   1. Run unit tests:        pytest tests/unit/ -v"
echo "   2. Check coverage:        pytest --cov=app --cov-report=html"
echo "   3. Run integration tests: pytest tests/integration/ --run-integration -v"
echo "   4. Read the guide:        cat tests/README.md"
echo ""
echo "📖 Full documentation: tests/README.md"
echo "=================================================="
