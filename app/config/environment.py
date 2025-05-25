from functools import lru_cache
from os import environ, path
from dotenv import load_dotenv

@lru_cache(maxsize=1)
def load_environment():
    """Load and cache environment variables"""
    # Load .env file only once
    load_dotenv()
    
    return {
        'APP_NAME': environ.get('APP_NAME') or 'flask-boilerplate',
        'APPLICATION_ENV': environ.get('APPLICATION_ENV') or 'development',
        'API_KEY': environ.get('API_KEY'),
        'GOOGLE_API_KEY': environ.get('GOOGLE_API_KEY'),
        'GOOGLE_PROJECT_ID': environ.get('GOOGLE_PROJECT_ID'),
        'GOOGLE_PROJECT_LOCATION': environ.get('GOOGLE_PROJECT_LOCATION'),
        'PORT': int(environ.get('PORT', 8080))
    }

def get_env(key):
    """Get environment variable by key"""
    return load_environment().get(key) 