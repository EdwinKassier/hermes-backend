# Hermes Backend Testing Suite

Comprehensive testing framework for the Hermes and Prism domains, with special focus on vector database integration and RAG quality assurance.

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Test Structure](#test-structure)
- [Running Tests](#running-tests)
- [Test Categories](#test-categories)
- [Configuration](#configuration)
- [CI/CD Integration](#cicd-integration)
- [Writing Tests](#writing-tests)
- [Coverage Reports](#coverage-reports)

## 🚀 Quick Start

### Installation

Install test dependencies:

```bash
# Install with test dependencies
pip install -e ".[test]"

# Or using uv (recommended)
uv pip install -e ".[test]"
```

### Run All Tests

```bash
# Run all unit tests
pytest tests/unit/ -v

# Run with coverage
pytest --cov=app --cov-report=html

# Run everything (unit + integration + performance)
pytest tests/ --run-integration --run-slow -v
```

## 📁 Test Structure

```
tests/
├── conftest.py                  # Main pytest configuration
├── fixtures/                    # Reusable test data
│   ├── models.py               # Domain model fixtures
│   └── __init__.py
├── utils/                       # Test utilities
│   ├── assertions.py           # Custom assertions
│   └── __init__.py
├── unit/                        # Fast, isolated unit tests
│   ├── hermes/
│   │   ├── test_models.py      # Hermes models
│   │   ├── test_services.py    # Hermes business logic
│   │   └── test_routes.py      # Hermes HTTP endpoints
│   └── prism/
│       ├── test_models.py      # Prism models
│       └── test_services.py    # Prism business logic
├── integration/                 # Integration tests
│   └── test_vector_db_integration.py  # ⭐ Vector DB + RAG tests
└── performance/                 # Performance tests
    └── test_vector_search_perf.py     # Performance benchmarks
```

## 🏃 Running Tests

### By Category

```bash
# Unit tests only (fast, no external dependencies)
pytest tests/unit/ -v

# Integration tests (requires real services)
pytest tests/integration/ --run-integration -v

# Performance tests (slow)
pytest tests/performance/ --run-slow -v
```

### By Domain

```bash
# Hermes domain tests
pytest tests/unit/hermes/ -v

# Prism domain tests
pytest tests/unit/prism/ -v

# Vector database tests
pytest tests/integration/test_vector_db_integration.py -v --run-integration
```

### By Marker

```bash
# Run only unit tests
pytest -m unit -v

# Skip slow tests
pytest -m "not slow" -v

# Run integration tests only
pytest -m integration --run-integration -v
```

### Specific Tests

```bash
# Run specific test file
pytest tests/unit/hermes/test_models.py -v

# Run specific test class
pytest tests/unit/hermes/test_models.py::TestUserIdentity -v

# Run specific test
pytest tests/unit/hermes/test_models.py::TestUserIdentity::test_user_identity_creation -v
```

### With Output Options

```bash
# Verbose output
pytest -vv

# Show print statements
pytest -s

# Stop on first failure
pytest -x

# Run last failed tests
pytest --lf

# Show slowest tests
pytest --durations=10
```

## 🎯 Test Categories

### Unit Tests (`tests/unit/`)

**Purpose**: Test individual components in isolation  
**Speed**: Fast (<1s per test)  
**Dependencies**: None (all mocked)

**Coverage**:
- ✅ Hermes models (UserIdentity, ConversationContext, etc.)
- ✅ Hermes services (process_request, chat, TTS generation)
- ✅ Hermes routes (HTTP endpoints, error handling)
- ✅ Prism models (PrismSession, TranscriptEntry, AudioChunk)
- ✅ Prism services (session management, bot orchestration, AI decisions)

**Example**:
```bash
pytest tests/unit/hermes/test_models.py -v
```

### Integration Tests (`tests/integration/`)

**Purpose**: Test interactions with real external services  
**Speed**: Moderate to slow (2-10s per test)  
**Dependencies**: Requires environment variables

**⭐ Critical Tests**:

#### Vector Database Integration (`test_vector_db_integration.py`)
- Tests live Supabase vector database queries
- Tests end-to-end RAG pipeline
- Tests Google Gemini embeddings generation
- Tests search quality, relevance, and accuracy

**Required Environment Variables**:
```bash
export GOOGLE_API_KEY="your-google-api-key"
export SUPABASE_URL="your-supabase-url"
export SUPABASE_SERVICE_ROLE_KEY="your-service-role-key"
```

#### TTS Audio Generation Integration (`test_tts_integration.py`)
- **ElevenLabs TTS**: Tests lowest latency audio generation (~75ms)
- **Google Cloud TTS**: Tests enterprise-grade voice synthesis
- **Chatterbox TTS**: Tests ML-based voice cloning (optional, requires heavy dependencies)
- Tests audio file generation, validation, and quality
- Tests multiple voices, languages, and customization
- Tests error handling and performance

**Required Environment Variables**:
```bash
# ElevenLabs (required for most TTS tests)
export EL_API_KEY="your-elevenlabs-api-key"

# Google Cloud TTS (required for Google TTS tests)
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/google-credentials.json"

# Chatterbox (optional - requires: pip install torch torchaudio chatterbox-tts)
# No API key needed, but tests are marked as @pytest.mark.slow
```

**Examples**:
```bash
# Run vector database tests
pytest tests/integration/test_vector_db_integration.py --run-integration -v

# Run TTS tests
pytest tests/integration/test_tts_integration.py --run-integration -v

# Run all integration tests
pytest tests/integration/ --run-integration -v

# Skip slow tests (like Chatterbox)
pytest tests/integration/ --run-integration -v -m "not slow"
```

### Performance Tests (`tests/performance/`)

**Purpose**: Benchmark performance and scalability  
**Speed**: Slow (10-60s per test)  
**Dependencies**: Requires real services

**Benchmarks**:
- Vector search latency (target: <2s avg)
- RAG pipeline latency (target: <10s avg)
- Embedding generation throughput
- Concurrent load handling

**Example**:
```bash
pytest tests/performance/ --run-integration --run-slow -v
```

## ⚙️ Configuration

### pytest.ini

Located at project root, configures:
- Test discovery patterns
- Coverage settings (70% minimum)
- Markers (unit, integration, slow)
- Warning filters

### conftest.py

Main configuration file with:
- Custom command-line options (`--run-integration`, `--run-slow`)
- Shared fixtures (mock services, sample data)
- Environment setup
- Test collection hooks

### Environment Variables

**Automatic .env File Loading** 🎉

The test suite **automatically loads** environment variables from `.env` file in the project root!

**Setup (Recommended)**:
```bash
# 1. Copy the example file
cp .env.test.example .env

# 2. Edit with your actual values
nano .env

# 3. Run tests - no manual exports needed!
pytest tests/integration/ --run-integration -v
```

**Required for Integration Tests**:
```bash
GOOGLE_API_KEY              # Google Gemini API key
SUPABASE_URL                # Supabase project URL
SUPABASE_SERVICE_ROLE_KEY   # Supabase service role key
```

**Optional**:
```bash
TEST_ENV                    # Test environment: local/ci/staging
APPLICATION_ENV             # Application environment (defaults to 'test')
```

**Alternative**: You can still export variables manually if preferred:
```bash
export GOOGLE_API_KEY="your-key"
export SUPABASE_URL="your-url"
export SUPABASE_SERVICE_ROLE_KEY="your-key"
```

## 🔄 CI/CD Integration

### GitHub Actions

Workflow file: `.github/workflows/test.yml`

**Jobs**:
1. **Unit Tests**: Runs on every push/PR
2. **Integration Tests**: Runs on main branch only
3. **Linting**: Code quality checks (flake8, black)
4. **Test Summary**: Aggregates results

**Triggers**:
- Push to `master`, `main`, or `develop`
- Pull requests to these branches

**Secrets Required** (in GitHub repo settings):
```
GOOGLE_API_KEY
SUPABASE_URL
SUPABASE_SERVICE_ROLE_KEY
```

### Local CI Simulation

```bash
# Run exactly what CI runs
pytest tests/unit/ -v --cov=app --cov-report=term-missing
flake8 app tests --count --select=E9,F63,F7,F82
black --check app tests
```

## ✍️ Writing Tests

### Test Naming Convention

```python
def test_<what>_<condition>_<expected>():
    """
    Test that <component> <action> when <condition>.
    Should <expected behavior>.
    """
```

**Examples**:
- `test_create_session_generates_user_id_when_not_provided`
- `test_handle_transcript_skip_bot_messages`
- `test_process_request_empty_text_raises_error`

### Using Fixtures

```python
def test_example(sample_user_identity, mock_gemini_service):
    """Use fixtures from conftest.py or fixtures/"""
    # Fixtures are automatically injected
    assert sample_user_identity.user_id is not None
```

### Custom Assertions

```python
from tests.utils.assertions import (
    assert_similarity_score_valid,
    assert_session_state_valid,
    assert_response_time_acceptable
)

def test_vector_search():
    results = search_vector_db("query")
    for doc, score in results:
        assert_similarity_score_valid(score)
```

### Mocking Services

```python
from unittest.mock import Mock, patch

def test_with_mocked_service():
    with patch('app.hermes.services.get_gemini_service') as mock:
        mock.return_value.generate_response.return_value = "Test response"
        # Your test code
```

### Parametrized Tests

```python
@pytest.mark.parametrize("input,expected", [
    ("hello", "HELLO"),
    ("world", "WORLD"),
])
def test_uppercase(input, expected):
    assert input.upper() == expected
```

## 📊 Coverage Reports

### Generate Coverage Report

```bash
# Terminal report
pytest --cov=app --cov-report=term-missing

# HTML report (opens in browser)
pytest --cov=app --cov-report=html
open htmlcov/index.html

# XML report (for CI/CD)
pytest --cov=app --cov-report=xml
```

### Coverage Goals

- **Minimum**: 70% overall (enforced in pytest.ini)
- **Target**: 80% overall
- **Critical paths**: 90%+ (RAG, authentication, core business logic)

### View Coverage

```bash
# After running tests with coverage
open htmlcov/index.html
```

## 🐛 Debugging Tests

### Run with Debug Output

```bash
# Show print statements
pytest -s tests/unit/hermes/test_services.py

# Very verbose
pytest -vv tests/unit/

# Show local variables on failure
pytest -l
```

### Run Single Test in Debug

```bash
# Use Python debugger
pytest --pdb tests/unit/hermes/test_services.py::test_name

# Or use ipdb (if installed)
pip install ipdb
pytest --pdb --pdbcls=IPython.terminal.debugger:Pdb
```

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'app'`  
**Solution**: Install package in editable mode: `pip install -e .`

**Issue**: Integration tests skipped  
**Solution**: Use `--run-integration` flag and set environment variables

**Issue**: "Vector store not initialized"  
**Solution**: Check `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` are set

## 📈 Performance Benchmarks

### Current Targets

| Operation | Target | Actual |
|-----------|--------|--------|
| Vector Search (avg) | <2s | Run tests to measure |
| Vector Search (P95) | <3s | Run tests to measure |
| RAG Pipeline (avg) | <10s | Run tests to measure |
| Embedding Generation | <5s | Run tests to measure |

### Run Benchmarks

```bash
pytest tests/performance/ --run-integration --run-slow -v
```

## 🔍 Test Quality Checklist

When writing tests, ensure:

- [ ] Test is independent (doesn't depend on other tests)
- [ ] Test is deterministic (same input → same output)
- [ ] Test has clear name describing what it tests
- [ ] Test has docstring explaining purpose
- [ ] Mocks are used for external dependencies (unit tests)
- [ ] Assertions are specific and meaningful
- [ ] Edge cases are covered
- [ ] Error cases are tested
- [ ] Test is fast (<1s for unit tests)

## 📚 Additional Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Python Testing Best Practices](https://docs.python-guide.org/writing/tests/)
- [Mocking with unittest.mock](https://docs.python.org/3/library/unittest.mock.html)

## 🤝 Contributing

When adding new features:

1. Write tests first (TDD approach)
2. Ensure all tests pass: `pytest tests/unit/ -v`
3. Check coverage: `pytest --cov=app`
4. Run linting: `flake8 app tests` and `black --check app tests`
5. Update this README if adding new test categories

## 📞 Support

For issues or questions:
1. Check test output for detailed error messages
2. Review test logs in `pytest` output
3. Check CI/CD logs in GitHub Actions
4. Refer to main project README for setup instructions

---

**Last Updated**: 2025-10-26  
**Test Coverage**: 70%+ (target)  
**Test Count**: 100+ tests across all categories

