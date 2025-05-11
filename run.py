
from datetime import datetime
import json
import os

from app import create_app


app = create_app()


@app.route('/status', methods=['GET'])
def status():
    return json.dumps({"message":  f'Hermes API Status : Running!'}), 200, {"ContentType": "application/json"}


@app.route('/', methods=['GET'])
def home():
    return json.dumps({"message": f'Welcome to the Hermes API'}), 200, {"ContentType": "application/json"}


if __name__ == '__main__':
    app.run(port=int(os.environ.get("PORT", 8080)),host='0.0.0.0',debug=True)
