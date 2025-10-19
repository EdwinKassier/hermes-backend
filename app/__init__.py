from flask import Flask, jsonify
from flask_cors import CORS

from .config import config as app_config
from .config.environment import get_env


def create_app():
    APPLICATION_ENV = get_env('APPLICATION_ENV') or 'development'
    app = Flask(get_env('APP_NAME') or 'hermes-backend')
    app.config.from_object(app_config[APPLICATION_ENV])

    # Enable CORS for API endpoints
    CORS(app, resources={r'/api/*': {'origins': '*'}})

    # Health check endpoint (required for Cloud Run)
    @app.route('/health', methods=['GET'])
    def health():
        return jsonify({
            'status': 'healthy',
            'service': 'hermes-backend',
            'environment': APPLICATION_ENV
        }), 200

    # Root endpoint
    @app.route('/', methods=['GET'])
    def home():
        return jsonify({
            'message': 'Welcome to Hermes API',
            'service': 'hermes-backend',
            'version': '1.0.0'
        }), 200

    # Status endpoint (from run.py)
    @app.route('/status', methods=['GET'])
    def status():
        return jsonify({
            'message': 'Hermes API Status: Running!',
            'status': 'operational'
        }), 200

    # Register Hermes API blueprint
    from .hermes import hermes
    app.register_blueprint(
        hermes,
        url_prefix='/api/v1/hermes'
    )

    # Register Prism API blueprint (Attendee voice agent integration)
    from .prism import prism_bp
    app.register_blueprint(prism_bp)

    return app
