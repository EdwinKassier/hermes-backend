from flask import Blueprint, current_app, request
from werkzeug.local import LocalProxy
import json
import glob
import os
import uuid
from google.cloud import dialogflowcx_v3beta1 as dialogflowcx
from authentication import check_auth

# Set up environment for credentials
file_pattern = "gha-creds-*.json"
file_matches = glob.glob(file_pattern)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
    file_matches[0]  # Or file_matches[0] if dynamically selected
)

# Flask Blueprint and logger
core = Blueprint("core", __name__)
logger = LocalProxy(lambda: current_app.logger)


@core.before_request
def before_request_func():
    current_app.logger.name = "core"


# Project and Agent IDs for Dialogflow CX
project_id = "edwin-portfolio-358212"
location = "us-central1"
agent_id = "72bae872-5c46-4cd9-8d3d-cfabf6b3f616"


# Function to create a session path
def create_session_path(session_id):
    return f"projects/{project_id}/locations/{location}/agents/{agent_id}/sessions/{session_id}"


def detect_intent_texts(agent, session_id, texts, language_code):
    session_path = f"{agent}/sessions/{session_id}"
    logger.info(f"Session path: {session_path}\n")
    client_options = None
    agent_components = dialogflowcx.AgentsClient.parse_agent_path(agent)
    location_id = agent_components["location"]

    if location_id != "global":
        api_endpoint = f"{location_id}-dialogflow.googleapis.com:443"
        logger.info(f"API Endpoint: {api_endpoint}\n")
        client_options = {"api_endpoint": api_endpoint}

    session_client = dialogflowcx.SessionsClient(client_options=client_options)

    fulfillment_texts = []
    for text in texts:
        text_input = dialogflowcx.TextInput(text=text)
        query_input = dialogflowcx.QueryInput(text=text_input, language_code=language_code)
        detect_intent_request = dialogflowcx.DetectIntentRequest(
            session=session_path, query_input=query_input
        )
        response = session_client.detect_intent(request=detect_intent_request)

        logger.info("=" * 20)
        logger.info(f"Query text: {response.query_result.text}")

        # Handle response messages correctly
        response_messages = [
            " ".join(msg.text.text) for msg in response.query_result.response_messages
        ]

        logger.info(f"Response text: {' '.join(response_messages)}\n")
        fulfillment_texts.append(" ".join(response_messages))

    return fulfillment_texts


# Route to process Dialogflow CX requests
@core.route("/process_request", methods=["GET", "POST"])
def process_request():
    logger.info("app test route hit")
    try:
        # Initialize agent path
        agent = f"projects/{project_id}/locations/{location}/agents/{agent_id}"
        
        # Generate unique session ID or use provided one
        session_id = request.args.get("session_id", str(uuid.uuid4()))
        
        # Get user input from request
        text_to_be_analyzed = str(request.args.get("request_text", "").strip())
        texts = [text_to_be_analyzed]
        language_code = "en-US"  # You can also get this from the request if needed
        
        # Call detect_intent_texts
        fulfillment_texts = detect_intent_texts(agent, session_id, texts, language_code)

        return (
            json.dumps({"message": " ".join(fulfillment_texts)}),
            200,
            {"Content-Type": "application/json"},
        )
    except Exception as e:
        logger.error(e)
        return json.dumps({"message": str(e)}), 500, {"Content-Type": "application/json"}


# Restricted route with authentication
@core.route("/restricted", methods=["GET"])
@check_auth
def restricted():
    return (
        json.dumps({"message": "Successful Auth"}),
        200,
        {"Content-Type": "application/json"},
    )


# Route to list files in the server directory
@core.route("/files", methods=["GET"])
def files():
    return (
        json.dumps({"message": str(os.listdir())}),
        200,
        {"Content-Type": "application/json"},
    )
