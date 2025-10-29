import multiprocessing
import os

# Server socket settings
bind = "0.0.0.0:8080"
backlog = 2048

# Worker processes
# Note: Using sync worker since WebSockets are now handled by separate server
# This prevents socket conflicts and ensures reliable HTTP response handling

# Determine provider for optimal worker configuration
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "elevenlabs").lower()

# ElevenLabs/Google: I/O-bound, low memory, thread-safe
# Use multiple workers with threading for optimal I/O handling
cpu_count = multiprocessing.cpu_count()
default_workers = cpu_count * 2  # 8 workers with 4 CPU
workers = int(os.getenv("GUNICORN_WORKERS", default_workers))
worker_connections = 100  # Reduced per-worker connections for better memory usage
threads = 4  # 4 threads per worker for I/O operations
print(
    f"✅ {TTS_PROVIDER.title()} mode: {workers} workers × {threads} threads × {worker_connections} connections"
)
print(
    f"   Theoretical capacity: {workers * threads * worker_connections} concurrent operations"
)
print(f"   Expected throughput: ~{workers * threads * 100} req/s")

# Use gthread worker for optimal I/O-bound operations (TTS requests)
worker_class = "gthread"
print("✅ Using gthread worker (optimized for I/O-bound TTS operations)")

# Timeouts
timeout = 120
keepalive = 65

# Preload app disabled to avoid DNS caching issues
preload_app = False

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"

# SSL
keyfile = None
certfile = None

# Process naming
proc_name = "hermes_backend"

# Server mechanics
daemon = False
pidfile = None
umask = 0
user = None
group = None
tmp_upload_dir = None

# Limits
max_requests = 10000
max_requests_jitter = 1000

# Worker recycling
graceful_timeout = 30
reload_extra_files = []


def post_fork(server, worker):
    server.log.info("Worker spawned (pid: %s)", worker.pid)


def pre_fork(server, worker):
    # No pre-fork setup needed
    pass


def pre_exec(server):
    server.log.info("Forked child, re-executing.")


def when_ready(server):
    server.log.info("Server is ready. Spawning workers")

    # Register signal handlers once per worker process (main thread)
    # This prevents the "signal only works in main thread" error
    import signal
    import threading

    def signal_handler(signum, frame):
        """Handle graceful shutdown signals."""
        server.log.info(f"Received signal {signum}, initiating graceful shutdown...")
        # The actual cleanup is handled by Gunicorn's graceful shutdown
        # This just ensures we don't get the threading error
        # signum and frame are required by signal handler signature

    # Only register if we're in the main thread of this process
    if threading.current_thread() is threading.main_thread():
        try:
            signal.signal(signal.SIGTERM, signal_handler)
            signal.signal(signal.SIGINT, signal_handler)
            server.log.info("Signal handlers registered successfully")
        except Exception as e:
            server.log.warning(f"Failed to register signal handlers: {e}")
    else:
        server.log.warning("Not in main thread, skipping signal registration")


def worker_int(worker):
    worker.log.info("worker received INT or QUIT signal")


def worker_abort(worker):
    worker.log.info("worker received SIGABRT signal")
