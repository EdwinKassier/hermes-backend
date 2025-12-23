"""
Pytest configuration and shared fixtures.
Main configuration file for the test suite.
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv

    env_file = project_root / ".env"
    if env_file.exists():
        load_dotenv(env_file)
        logging.info(f"✓ Loaded environment variables from {env_file}")
except ImportError:
    logging.warning("python-dotenv not installed, skipping .env file loading")
except Exception as e:
    logging.warning(f"Failed to load .env file: {e}")

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def pytest_addoption(parser):
    """Add custom command line options"""
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests (requires real services)",
    )
    parser.addoption(
        "--run-slow", action="store_true", default=False, help="Run slow tests"
    )
    parser.addoption(
        "--vector-db-url",
        action="store",
        default=None,
        help="Vector database URL for integration tests",
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on options"""
    # Skip integration tests unless --run-integration
    if not config.getoption("--run-integration"):
        skip_integration = pytest.mark.skip(
            reason="need --run-integration option to run"
        )
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)

    # Skip slow tests unless --run-slow
    if not config.getoption("--run-slow"):
        skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line(
        "markers", "requires_api_key: marks tests that require real API keys"
    )


# ==================== Environment & Configuration ====================


@pytest.fixture(scope="session")
def test_env():
    """Setup test environment variables"""
    os.environ.setdefault("APPLICATION_ENV", "test")
    os.environ.setdefault("API_KEY", "test_api_key")
    yield


@pytest.fixture(scope="session")
def test_environment():
    """Determine test environment (local/ci/staging)"""
    return os.environ.get("TEST_ENV", "local")


@pytest.fixture(scope="session")
def integration_config(test_environment):
    """Configuration for integration tests"""
    # Try multiple environment variable names for flexibility
    supabase_url = os.environ.get("SUPABASE_URL") or os.environ.get(
        "SUPABASE_PROJECT_URL"
    )

    configs = {
        "local": {
            "supabase_url": supabase_url,
            "google_api_key": os.environ.get("GOOGLE_API_KEY"),
            "supabase_service_key": os.environ.get("SUPABASE_SERVICE_ROLE_KEY"),
            "timeout": 30,
            "retry_attempts": 3,
        },
        "ci": {
            "supabase_url": os.environ.get("CI_SUPABASE_URL"),
            "google_api_key": os.environ.get("GOOGLE_API_KEY"),
            "supabase_service_key": os.environ.get("CI_SUPABASE_SERVICE_ROLE_KEY"),
            "timeout": 60,
            "retry_attempts": 5,
        },
        "staging": {
            "supabase_url": os.environ.get("STAGING_SUPABASE_URL"),
            "google_api_key": os.environ.get("GOOGLE_API_KEY"),
            "supabase_service_key": os.environ.get("STAGING_SUPABASE_SERVICE_ROLE_KEY"),
            "timeout": 45,
            "retry_attempts": 3,
        },
    }
    return configs.get(test_environment, configs["local"])


# ==================== Mock Services ====================


@pytest.fixture
def mock_redis():
    """Mock Redis for unit tests"""
    mock = Mock()
    mock.get.return_value = None
    mock.set.return_value = True
    mock.delete.return_value = True
    mock.exists.return_value = False
    return mock


@pytest.fixture
def mock_llm_service():
    """Mock LLMService."""
    mock = MagicMock()
    mock.generate_response.return_value = "Mock response"
    return mock


@pytest.fixture
def mock_tts_service():
    """Mock TTS service"""
    mock = Mock()
    mock.generate_audio.return_value = {
        "cloud_url": "https://storage.example.com/audio.wav",
        "local_path": "/tmp/audio.wav",
        "audio_format": "wav",
    }
    mock.tts_provider = "test_provider"
    return mock


# ==================== Sample Data ====================


@pytest.fixture
def sample_audio_data():
    """Provide sample audio data for testing"""
    # Generate simple PCM audio (1 second of silence)
    sample_rate = 16000
    duration = 1.0
    num_samples = int(sample_rate * duration)

    # 16-bit PCM silence
    audio_data = b"\x00\x00" * num_samples
    return audio_data


@pytest.fixture
def sample_audio_pcm():
    """Generate sample PCM audio data with sine wave"""
    try:
        import numpy as np

        # 16kHz, 16-bit mono PCM, 1 second
        sample_rate = 16000
        duration = 1.0
        num_samples = int(sample_rate * duration)

        # Generate sine wave at 440Hz (A4 note)
        t = np.linspace(0, duration, num_samples)
        frequency = 440
        audio = np.sin(2 * np.pi * frequency * t)

        # Convert to 16-bit PCM
        audio_int16 = (audio * 32767).astype(np.int16)
        return audio_int16.tobytes()
    except ImportError:
        # Fallback if numpy not available
        return b"\x00\x00" * 16000


@pytest.fixture
def sample_transcript():
    """Provide sample transcript for testing"""
    return {
        "speaker": "John Doe",
        "text": "Hello, this is a test transcript",
        "is_final": True,
        "timestamp": datetime.utcnow().isoformat(),
    }


@pytest.fixture
def sample_meeting_url():
    """Provide sample Google Meet URL"""
    return "https://meet.google.com/abc-defg-hij"


# ==================== Logging ====================


@pytest.fixture(autouse=True)
def setup_test_logging(caplog):
    """Configure logging for tests"""
    caplog.set_level(logging.INFO)


# ==================== Cleanup ====================


@pytest.fixture(autouse=True)
def cleanup_temp_files():
    """Clean up temporary files after each test"""
    import shutil
    import tempfile

    # Create temp directory for test
    temp_dir = tempfile.mkdtemp(prefix="hermes_test_")

    yield temp_dir

    # Cleanup after test
    try:
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception as e:
        logging.warning(f"Failed to cleanup temp directory: {e}")
