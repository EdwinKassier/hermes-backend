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

if TTS_PROVIDER == "chatterbox":
    # Chatterbox: CPU-bound, high memory, NOT thread-safe
    # Use single worker to prevent:
    # - Model state corruption (race conditions)
    # - Memory exhaustion (2.5GB per worker)
    # - CPU contention (GIL thrashing)
    workers = 1
    worker_connections = 50  # Lower for CPU-bound work
    print("⚠️  Chatterbox mode: Single worker (NOT suitable for high scale)")
    print("    Recommendation: Use ElevenLabs or Google for production scale")

else:
    # ElevenLabs/Google: I/O-bound, low memory, thread-safe
    # Use multiple workers for true parallelism and fault isolation
    cpu_count = multiprocessing.cpu_count()
    default_workers = min(cpu_count * 2, 8)  # 2x CPU cores, max 8
    workers = int(os.getenv("GUNICORN_WORKERS", default_workers))
    worker_connections = 1000
    print(
        f"✅ {TTS_PROVIDER.title()} mode: {workers} workers × {worker_connections} connections"
    )
    print(
        f"   Theoretical capacity: {workers * worker_connections} concurrent connections"
    )
    print(f"   Expected throughput: ~{workers * 100} req/s")

# Use sync worker for reliable HTTP handling (WebSockets handled by separate server)
worker_class = "sync"
print("✅ Using sync worker (WebSockets handled by separate server)")

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
    pass


def pre_exec(server):
    server.log.info("Forked child, re-executing.")


def when_ready(server):
    server.log.info("Server is ready. Spawning workers")


def worker_int(worker):
    worker.log.info("worker received INT or QUIT signal")


def worker_abort(worker):
    worker.log.info("worker received SIGABRT signal")
