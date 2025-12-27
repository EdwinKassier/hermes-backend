"""Hermes API Routes - HTTP endpoints following DDD architecture."""

import os

from flask import Blueprint, current_app, g, jsonify, request
from pydantic import ValidationError
from werkzeug.local import LocalProxy

from app.shared.services.IdentityService import IdentityService
from authentication import check_auth

from .constants import (
    ERROR_REQUEST_BODY_REQUIRED,
    ERROR_UNEXPECTED,
    ERROR_VALIDATION_FAILED,
    HTTP_BAD_REQUEST,
    HTTP_INTERNAL_ERROR,
    HTTP_OK,
    SUCCESS_AUTH,
    SUCCESS_CHAT_PROCESSED,
    SUCCESS_CONTEXT_CLEARED,
    SUCCESS_REQUEST_PROCESSED,
    SUCCESS_VECTOR_SYNC,
)
from .exceptions import HermesError, InvalidRequestError, VectorSyncError
from .models import ResponseMode, UserIdentity
from .schemas import (
    AuthResponseSchema,
    ChatMessageSchema,
    ChatResponseSchema,
    ErrorResponseSchema,
    FileListResponseSchema,
    HealthCheckResponseSchema,
    ProcessRequestResponseSchema,
    ProcessRequestSchema,
    VectorSyncRequestSchema,
    VectorSyncResponseSchema,
)
from .services import get_hermes_service

# Flask Blueprint and logger
hermes = Blueprint("hermes", __name__)
logger = LocalProxy(lambda: current_app.logger)


@hermes.before_request
def before_request_func():
    """Process before each request to set up request context."""
    current_app.logger.name = "hermes"

    # Store identity information in the request context
    g.identity = IdentityService.get_identity_fingerprint(request)
    current_app.logger.debug(f"Request from user: {g.identity}")


@hermes.errorhandler(HermesError)
def handle_hermes_error(error: HermesError):
    """Handle domain-specific errors."""
    logger.error(f"Hermes error: {error.message}")
    response = ErrorResponseSchema(
        error=error.code, message=error.message, details=error.details
    )
    return jsonify(response.model_dump()), HTTP_BAD_REQUEST


@hermes.errorhandler(ValidationError)
def handle_validation_error(error: ValidationError):
    """Handle Pydantic validation errors."""
    logger.error(f"Validation error: {error}")
    response = ErrorResponseSchema(
        error="VALIDATION_ERROR",
        message=ERROR_VALIDATION_FAILED,
        details={"errors": error.errors()},
    )
    return jsonify(response.model_dump()), HTTP_BAD_REQUEST


@hermes.errorhandler(Exception)
def handle_generic_error(error: Exception):
    """Handle unexpected errors."""
    logger.error(f"Unexpected error: {error}", exc_info=True)
    response = ErrorResponseSchema(
        error="INTERNAL_ERROR", message=ERROR_UNEXPECTED, details={"error": str(error)}
    )
    return jsonify(response.model_dump()), HTTP_INTERNAL_ERROR


@hermes.route("/process_request", methods=["GET", "POST"])
def process_request():
    """
    Process user requests through Gemini with RAG.

    Supports both GET (query parameters) and POST (JSON body) methods.

    Query Parameters (GET or POST):
        request_text (str): Text to process
        response_mode (str): Response mode (text or tts)
        persona (str): AI persona to use ('hermes' or 'prisma')
        legion_mode (bool/str): Whether to use legion processing mode (default: false)

    Request Body (POST with JSON):
        {
            "request_text": "Text to process",
            "response_mode": "text",
            "persona": "hermes",
            "legion_mode": false
        }

    Returns:
        JSON response with AI-generated content
    """
    logger.info("Process request route hit")

    # Handle both JSON body (POST) and query params (GET/POST)
    if request.method == "POST" and request.is_json:
        data = request.get_json() or {}
        params = ProcessRequestSchema(
            request_text=data.get("request_text", ""),
            response_mode=data.get("response_mode", "text"),
            persona=data.get("persona", "hermes"),
            legion_mode=data.get("legion_mode", False),
        )
    else:
        # Handle query parameters (GET or POST without JSON)
        params = ProcessRequestSchema(
            request_text=request.args.get("request_text", ""),
            response_mode=request.args.get("response_mode", "text"),
            persona=request.args.get("persona", "hermes"),
            legion_mode=request.args.get("legion_mode", False),
        )

    # Create user identity model
    user_identity = UserIdentity(**g.identity)

    # Get service and process request
    service = get_hermes_service()
    result = service.process_request(
        text=params.request_text,
        user_identity=user_identity,
        response_mode=ResponseMode(params.response_mode),
        persona=params.persona,
        legion_mode=params.legion_mode,
    )

    # Build response
    response = ProcessRequestResponseSchema(
        message=result.message,
        response_mode=result.response_mode.value,
        wave_url=result.audio_url,
        tts_provider=result.tts_provider,
        user_id=result.user_id,
        timestamp=result.timestamp,
        metadata=result.metadata,
    )

    logger.info(SUCCESS_REQUEST_PROCESSED)
    return jsonify(response.model_dump()), HTTP_OK


@hermes.route("/chat", methods=["POST"])
def chat():
    """
    Chat endpoint for conversational interactions.

    Request Body:
        {
            "message": "User's chat message",
            "include_context": true,
            "persona": "hermes",
            "legion_mode": false
        }

    Returns:
        JSON response with AI reply
    """
    logger.info("Chat route hit")

    # Validate request body
    data = request.get_json()
    if not data:
        raise InvalidRequestError(ERROR_REQUEST_BODY_REQUIRED)

    chat_request = ChatMessageSchema(**data)

    # Create user identity model
    user_identity = UserIdentity(**g.identity)

    # Get service and process chat
    service = get_hermes_service()
    result = service.chat(
        message=chat_request.message,
        user_identity=user_identity,
        include_context=chat_request.include_context,
        persona=chat_request.persona,
        legion_mode=chat_request.legion_mode,
    )

    # Build response
    response = ChatResponseSchema(
        message=result.content,
        user_id=result.user_id,
        timestamp=result.timestamp,
        metadata=result.metadata,
    )

    logger.info(SUCCESS_CHAT_PROCESSED)
    return jsonify(response.model_dump()), HTTP_OK


@hermes.route("/restricted", methods=["GET"])
@check_auth
def restricted():
    """
    Protected endpoint requiring authentication.

    Returns:
        JSON response confirming authentication
    """
    response = AuthResponseSchema(message=SUCCESS_AUTH, authenticated=True)
    return jsonify(response.model_dump()), HTTP_OK


@hermes.route("/files", methods=["GET"])
def files():
    """
    List files in the server directory (for debugging).

    Returns:
        JSON response with file list
    """
    file_list = os.listdir()
    response = FileListResponseSchema(files=file_list, count=len(file_list))
    return jsonify(response.model_dump()), HTTP_OK


@hermes.route("/health", methods=["GET"])
def health_check():
    """
    Health check endpoint for the Hermes service.

    Returns:
        JSON response with service health status
    """
    response = HealthCheckResponseSchema(
        status="healthy", service="hermes", message="Hermes API is running"
    )
    return jsonify(response.model_dump()), HTTP_OK


@hermes.route("/clear-context", methods=["POST"])
def clear_context():
    """
    Clear conversation context for the current user.

    Returns:
        JSON response confirming context cleared
    """
    user_identity = UserIdentity(**g.identity)
    service = get_hermes_service()
    service.clear_conversation_context(user_identity.user_id)

    return (
        jsonify({"message": SUCCESS_CONTEXT_CLEARED, "user_id": user_identity.user_id}),
        HTTP_OK,
    )


@hermes.route("/sync-vectors", methods=["POST"])
def sync_vectors():
    """
    Sync vector store with latest documents from GCS bucket.

    This endpoint triggers a synchronization operation that:
    1. Downloads documents from the configured GCS bucket
    2. Processes and chunks the documents (with Gemini OCR for PDFs)
    3. Generates embeddings using Gemini API
    4. Upserts embeddings to Supabase vector store

    Request Body:
        {
            "bucket_name": "optional-bucket-name",
            "folder_path": "optional/path",
            "force_refresh": false
        }

    Returns:
        JSON response with sync results including:
        - status: Operation status (completed, completed_with_errors)
        - documents_processed: Number of documents processed
        - chunks_generated: Number of chunks created
        - embeddings_created: Number of embeddings generated
        - duration_seconds: Total operation time
        - errors: List of any errors encountered
    """
    logger.info("Vector sync route hit")

    # Import service here to avoid circular imports
    from .vector_sync_service import get_vector_sync_service

    # Parse request body (all fields are optional)
    data = request.get_json() or {}
    sync_request = VectorSyncRequestSchema(**data)

    try:
        # Get service and perform sync
        service = get_vector_sync_service()
        result = service.sync_vectors(
            bucket_name=sync_request.bucket_name,
            folder_path=sync_request.folder_path or "",
            force_refresh=sync_request.force_refresh,
        )

        # Build response
        response = VectorSyncResponseSchema(
            status=result.status,
            documents_processed=result.documents_processed,
            chunks_generated=result.chunks_generated,
            embeddings_created=result.embeddings_created,
            duration_seconds=result.duration_seconds,
            errors=result.errors,
            timestamp=result.timestamp,
        )

        logger.info(SUCCESS_VECTOR_SYNC)
        return jsonify(response.model_dump()), HTTP_OK

    except VectorSyncError as e:
        logger.error(f"Vector sync error: {e.message}")
        response = ErrorResponseSchema(
            error=e.code, message=e.message, details=e.details
        )
        return jsonify(response.model_dump()), HTTP_INTERNAL_ERROR
