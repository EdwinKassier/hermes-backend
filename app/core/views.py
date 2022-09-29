
from flask import Blueprint, current_app,request
from werkzeug.local import LocalProxy
import json

from authentication import check_auth

import os
from google.cloud import dialogflow_v2

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "../../credentials.json"

#os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "D:\hermes-backend\credentials.json"

session_client = dialogflow_v2.SessionsClient()
session = session_client.session_path("edwin-portfolio-358212", "Test")


from .tasks import test_task

core = Blueprint('core', __name__)
logger = LocalProxy(lambda: current_app.logger)


@core.before_request
def before_request_func():
    current_app.logger.name = 'core'

#Preparing for prod release cloud run, test
@core.route('/process_request', methods=['GET'])
def test():
    logger.info('app test route hit')
    try:

        text_to_be_analyzed = str(request.args.get('request_text').strip())

        text_input = dialogflow_v2.TextInput(text=text_to_be_analyzed, language_code='en')

        query_input = dialogflow_v2.QueryInput(text=text_input)

        response = session_client.detect_intent(
            request={"session": session, "query_input": query_input}
        )

        print("=" * 20)
        print("Query text: {}".format(response.query_result.query_text))
        print(
            "Detected intent: {} (confidence: {})\n".format(
                response.query_result.intent.display_name,
                response.query_result.intent_detection_confidence,
            )
        )
        print("Fulfillment text: {}\n".format(response.query_result.fulfillment_text))

        return json.dumps({"message": response.query_result.fulfillment_text}), 200, {"ContentType": "application/json"}
    except Exception as e:
        print(e)
        return json.dumps({"message": e}), 500, {"ContentType": "application/json"}



@core.route('/restricted', methods=['GET'])
@check_auth
def restricted():
    return json.dumps({"message": 'Successful Auth'}), 200, {"ContentType": "application/json"}
