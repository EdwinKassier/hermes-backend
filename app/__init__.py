import logging.config
from celery import Celery
from flask import Flask
from flask_cors import CORS

from .config import config as app_config
from .config.environment import get_env

celery = Celery(__name__)

#Called in the run function, this creates and links the main app logic to the exposed endpoint
def create_app():
    APPLICATION_ENV = get_env('APPLICATION_ENV')
    app = Flask(get_env('APP_NAME'))
    app.config.from_object(app_config[APPLICATION_ENV])

    #This step is critical for allowing testing while in a local environment
    CORS(app, resources={r'/api/*': {'origins': '*'}})

    celery.config_from_object(app.config, force=True)
    
    # We are registering the core folder as a blueprint, this allows us to more easily create other blueprints(ie endpoints) in future if needed, say api/sweepsouth/support
    #This allows for easier extension in future
    from .core.views import core as core_blueprint
    app.register_blueprint(
        core_blueprint,
        url_prefix='/api/v1/project/core'
    )

    return app
