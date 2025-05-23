from flask import Blueprint, current_app, request
from werkzeug.local import LocalProxy
import json
import glob
import os
import uuid
from google.cloud import dialogflowcx_v3beta1 as dialogflowcx
from authentication import check_auth

from ..services.GeminiService import GeminiService 

# Flask Blueprint and logger
core = Blueprint("core", __name__)
logger = LocalProxy(lambda: current_app.logger)

gemini = GeminiService()


@core.before_request
def before_request_func():
    current_app.logger.name = "core"

# Route to process Dialogflow CX requests
@core.route("/process_request", methods=["GET", "POST"])
def process_request():
    logger.info("app test route hit")
    try:
        # Get user input from request
        text_to_be_analyzed = str(request.args.get("request_text", "").strip())

        result = gemini.generate_gemini_response_with_rag(text_to_be_analyzed)
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
