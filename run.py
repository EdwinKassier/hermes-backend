# IMPORTANT: Gevent monkey patching MUST be first, before any other imports
# This prevents RecursionError when using requests/urllib3 with gevent

# Note: GEVENT_RESOLVER=dnspython must be set in environment BEFORE Python starts
# This is handled in scripts/dev_with_ngrok.sh

import gevent.monkey
gevent.monkey.patch_all()

from app import create_app
from app.config.environment import get_env

app = create_app()

if __name__ == '__main__':
    port = get_env('PORT')
    debug = get_env('APPLICATION_ENV') != 'production'
    app.run(port=port, host='0.0.0.0', debug=debug)
