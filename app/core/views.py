from flask import Blueprint, current_app, request, g
from werkzeug.local import LocalProxy
import json
import glob
import os
import uuid
from google.cloud import dialogflowcx_v3beta1 as dialogflowcx
from authentication import check_auth
from app.utils.service_loader import get_gemini_service, get_tts_service
from app.services import IdentityService

# Flask Blueprint and logger
core = Blueprint("core", __name__)
logger = LocalProxy(lambda: current_app.logger)

# Remove global gemini instance
# gemini = GeminiService()

@core.before_request
def before_request_func():
    """Process before each request to set up request context.
    
    - Sets the logger name
    - Captures and stores identity information in the request context
    """
    current_app.logger.name = "core"
    
    # Store identity information in the request context
    g.identity = IdentityService.get_identity_fingerprint(request)
    current_app.logger.debug(f"Request from user: {g.identity}")

# Route to process Dialogflow CX requests
@core.route("/process_request", methods=["GET", "POST"])
def process_request():
    logger.info("app test route hit")
    try:
        # Get user input from request
        text_to_be_analyzed = str(request.args.get("request_text", "").strip())
        response_mode = str(request.args.get("response_mode", "text").strip())

        # Get user identity from request context
        user_identity = g.identity['user_id']
        
        gemini = get_gemini_service()
        result = gemini.generate_gemini_response_with_rag(
            prompt=text_to_be_analyzed,
            user_id=user_identity
        )
        
        if response_mode == "tts":
            tts = get_tts_service()
            tts_result = tts.generate_audio(result)
            return (
                json.dumps({"message": result, "wave_url": tts_result['cloud_url']}),
                200,
                {"Content-Type": "application/json"},
            )
        else:
            return (
                json.dumps({"message": result}),
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
# Update any routes that use gemini to use get_gemini_service() instead
# For example:
@core.route("/chat", methods=["POST"])
def chat():
    gemini = get_gemini_service()
    return json.dumps({"message": "Chat endpoint"}), 200, {"Content-Type": "application/json"}

@core.route('/health', methods=['GET'])
def status():
    return json.dumps({"message": f'Hermes API Status : Running!'}), 200, {"ContentType": "application/json"}
