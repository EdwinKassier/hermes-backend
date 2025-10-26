from flask import Flask, jsonify
from flask_cors import CORS
import os
import logging
from .config import config as app_config


def create_app():
    # Set default environment to development if not specified
    APPLICATION_ENV = os.getenv('APPLICATION_ENV', 'development')
    
    # Initialize Flask app
    app = Flask(__name__)
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    app.config.from_object(app_config[APPLICATION_ENV])

    # Enable CORS for API endpoints
    CORS(app, resources={r'/api/*': {'origins': '*'}})

    # Health check endpoint (required for Cloud Run and health checks)
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
            'version': '1.0.0',
            'environment': APPLICATION_ENV
        }), 200

    # Status endpoint
    @app.route('/status', methods=['GET'])
    def status():
        return jsonify({
            'message': 'Hermes API Status: Running!',
            'status': 'operational',
            'environment': APPLICATION_ENV
        }), 200

    try:
        # Register Hermes API blueprint
        from .hermes.routes import hermes
        app.register_blueprint(
            hermes,
            url_prefix='/api/v1/hermes'
        )
        logger.info("Successfully registered Hermes blueprint")
    except ImportError as e:
        logger.warning(f"Failed to register Hermes blueprint: {e}")

    try:
        # Register Prism API blueprint (Attendee voice agent integration)
        from .prism.routes import prism_bp
        app.register_blueprint(
            prism_bp,
            url_prefix='/api/v1/prism'
        )
        logger.info("Successfully registered Prism blueprint")
    except ImportError as e:
        logger.warning(f"Failed to register Prism blueprint: {e}")

    return app
