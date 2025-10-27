from app import create_app
from app.config.environment import get_env

app = create_app()

if __name__ == "__main__":
    port = get_env("PORT")
    debug = get_env("APPLICATION_ENV") != "production"
    app.run(port=port, host="0.0.0.0", debug=debug)
