# CRITICAL: Gevent monkey patching MUST be first
# Only import and patch if gevent is available
try:
    import gevent.monkey
    gevent.monkey.patch_all()
    GEVENT_AVAILABLE = True
except ImportError:
    GEVENT_AVAILABLE = False
    print("WARNING: gevent not available, falling back to sync worker")

import multiprocessing

# Server socket settings
bind = "0.0.0.0:8080"
backlog = 2048

# Worker processes
# Note: Using gevent for WebSocket support (simple-websocket compatible)
# Fallback to sync worker if gevent not available
workers = 1  # Single worker for WebSocket session affinity
worker_class = "gevent" if GEVENT_AVAILABLE else "sync"
worker_connections = 1000 if GEVENT_AVAILABLE else 40

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