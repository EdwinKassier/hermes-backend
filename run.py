from datetime import datetime
import json

from app import create_app
from app.config.environment import get_env

app = create_app()

@app.route('/status', methods=['GET'])
def status():
    return json.dumps({"message": f'Hermes API Status : Running!'}), 200, {"ContentType": "application/json"}

@app.route('/', methods=['GET'])
def home():
    return json.dumps({"message": f'Welcome to the Hermes API'}), 200, {"ContentType": "application/json"}

if __name__ == '__main__':
    port = get_env('PORT')
    app.run(port=port, host='0.0.0.0', debug=True)
