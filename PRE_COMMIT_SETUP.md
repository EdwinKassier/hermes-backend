# Pre-commit Hooks Setup

This project uses pre-commit hooks to ensure code quality and consistency before commits reach the CI/CD pipeline.

## What's Included

### Code Formatting & Quality
- **Black**: Python code formatting (line length: 88)
- **isort**: Import sorting (compatible with Black)
- **flake8**: Linting with sensible ignores for common patterns
- **pylint**: Static analysis (matching CI/CD pipeline settings)

### File Hygiene
- **trailing-whitespace**: Remove trailing whitespace
- **end-of-file-fixer**: Ensure files end with newline
- **check-yaml/json/toml**: Validate configuration files
- **check-merge-conflict**: Detect merge conflict markers
- **check-added-large-files**: Prevent large files (>1MB)
- **check-ast**: Python syntax validation
- **debug-statements**: Detect debug statements
- **detect-private-key**: Security check for private keys
- **check-case-conflict**: Detect case-sensitive filename conflicts

### Testing
- **pytest-check**: Run tests on pre-push (not every commit)

## Installation

The hooks are already installed. If you need to reinstall:

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install --install-hooks

# Install pre-push hook for tests
pre-commit install --hook-type pre-push
```

## Usage

### Automatic (Recommended)
Hooks run automatically on:
- `git commit` (formatting, linting, file checks)
- `git push` (includes test run)

### Manual
```bash
# Run on all files
pre-commit run --all-files

# Run on staged files only
pre-commit run

# Run specific hook
pre-commit run black
pre-commit run flake8
```

## Configuration

- **`.pre-commit-config.yaml`**: Main configuration
- **`.flake8`**: flake8-specific settings
- **`pyproject.toml`**: Black and isort settings

## Benefits

1. **Early Detection**: Catch issues before CI/CD
2. **Consistency**: Enforce code style across team
3. **Pipeline Alignment**: Same checks as CI/CD
4. **Developer Experience**: Fast feedback loop
5. **Quality Assurance**: Prevent common mistakes

## Troubleshooting

### Skip Hooks (Emergency Only)
```bash
git commit --no-verify
git push --no-verify
```

### Update Hook Versions
```bash
pre-commit autoupdate
```

### Clear Cache
```bash
pre-commit clean
```

## CI/CD Integration

These hooks mirror the checks in `.github/workflows/push.yml`:
- ✅ Black formatting
- ✅ isort import sorting
- ✅ flake8 linting
- ✅ pylint static analysis
- ✅ File hygiene checks

The pipeline will pass if pre-commit hooks pass locally.
