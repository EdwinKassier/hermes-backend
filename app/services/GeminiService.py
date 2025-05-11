import os
import traceback
import io  # For handling bytes streams with pypdf
from typing import List, Dict, Any, Tuple, Optional

# Google / Gemini / Langchain specific
from langchain_google_genai import ChatGoogleGenerativeAI  # For the LLM
from langchain_google_vertexai import VertexAIEmbeddings  # Changed for embeddings

# from google.ai.generativelanguage_v1beta.types import Tool as GenAITool # Not used
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import google.generativeai as genai
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document

# Text processing and vector stores
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
)  # Older import path, but still works

# from langchain_text_splitters import RecursiveCharacterTextSplitter # Newer import path
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore


# PDF and GCS
from google.cloud import storage
import pypdf

import logging

# Setup basic logging configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class GeminiService:
    MODEL_NAME = "gemini-2.0-flash"  # LLM for generation
    TEMPERATURE = 0.6
    TIMEOUT = 60
    MAX_RETRIES = 3
    ERROR_MESSAGE = "Sorry, the Hermes LLM service encountered an error."

    # Changed to a Vertex AI embedding model
    EMBEDDING_MODEL_NAME = "text-embedding-004"
    TEXT_SPLITTER_CHUNK_SIZE = 1500
    TEXT_SPLITTER_CHUNK_OVERLAP = 250
    RAG_TOP_K = 3

    SAFETY_SETTINGS = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    def __init__(
        self,
        initial_gcs_bucket_name: Optional[str] = "ashes-project-hermes-training",
        initial_gcs_folder_path: Optional[str] = None,
        gcs_credentials_path: Optional[str] = "credentials.json",
        vertex_project: Optional[str] = os.environ["GOOGLE_PROJECT_ID"],
        vertex_location: Optional[str] = os.environ["GOOGLE_PROJECT_LOCATION"],
    ):
        # For Gemini LLM (ChatGoogleGenerativeAI)
        if "GOOGLE_API_KEY" not in os.environ:
            raise ValueError(
                "GOOGLE_API_KEY environment variable not set for Gemini LLM."
            )
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./credentials.json"

        self.model = ChatGoogleGenerativeAI(
            model=self.MODEL_NAME,
            temperature=self.TEMPERATURE,
            max_retries=self.MAX_RETRIES,
            safety_settings=self.SAFETY_SETTINGS,
            request_timeout=self.TIMEOUT,
        )
        self.base_prompt = os.environ.get("BASE_PROMPT", "")

        # Initialize GCS Client
        try:
            if gcs_credentials_path and os.path.exists(gcs_credentials_path):
                self.storage_client = storage.Client.from_service_account_json(
                    gcs_credentials_path
                )
                logging.info(
                    f"Initialized Google Cloud Storage client using credentials from: {gcs_credentials_path}"
                )
            else:
                if gcs_credentials_path:
                    logging.warning(
                        f"GCS credentials file not found at '{gcs_credentials_path}'. "
                        "Attempting to use Application Default Credentials for GCS."
                    )
                else:
                    logging.info(
                        "No GCS credentials file path provided. "
                        "Attempting to use Application Default Credentials for GCS."
                    )
                self.storage_client = storage.Client()
                logging.info(
                    "Initialized Google Cloud Storage client using Application Default Credentials."
                )
        except Exception as e:
            logging.error(
                f"Failed to initialize Google Cloud Storage client: {e}. "
                "Ensure GCS credentials (file or ADC) are correctly set up."
            )
            raise

        # Initialize VertexAIEmbeddings
        try:
            self.embeddings_model = VertexAIEmbeddings(
                model_name=self.EMBEDDING_MODEL_NAME,
                project=vertex_project,  # Pass if ADC needs help or for explicit project
                location=vertex_location,  # Pass if ADC needs help or for explicit location
            )
            logging.info(
                f"Initialized VertexAIEmbeddings with model: {self.EMBEDDING_MODEL_NAME}. "
                f"Project: {vertex_project or 'default from ADC'}, Location: {vertex_location or 'default from ADC'}."
            )
        except Exception as e:
            logging.error(
                f"Failed to initialize VertexAIEmbeddings: {e}. "
                "Ensure Application Default Credentials for Vertex AI are set up "
                "or GOOGLE_APPLICATION_CREDENTIALS env var points to a valid service account key "
                "with Vertex AI User role."
            )
            raise

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.TEXT_SPLITTER_CHUNK_SIZE,
            chunk_overlap=self.TEXT_SPLITTER_CHUNK_OVERLAP,
        )

        self.gcs_bucket_name: Optional[str] = None
        self.gcs_folder_path: Optional[str] = None
        self.vector_store: Optional[VectorStore] = (
            None  # Using base class for type hint
        )

        if initial_gcs_bucket_name:
            self.update_rag_sources(initial_gcs_bucket_name, initial_gcs_folder_path)

        logging.info(
            f"GeminiService core initialized. LLM: {self.MODEL_NAME}, Embeddings: {self.EMBEDDING_MODEL_NAME} (Vertex AI)"
        )
        if self.vector_store:
            logging.info(
                f"Initial RAG sources processed from gs://{self.gcs_bucket_name}/{self.gcs_folder_path or ''}."
            )
        else:
            logging.info("No initial RAG sources provided or processed at startup.")

    def update_rag_sources(
        self, gcs_bucket_name: str, gcs_folder_path: Optional[str] = None
    ):
        logging.info(
            f"Updating RAG sources from GCS bucket: gs://{gcs_bucket_name}/{gcs_folder_path or ''}"
        )
        self.gcs_bucket_name = gcs_bucket_name
        if gcs_folder_path and not gcs_folder_path.endswith("/"):
            self.gcs_folder_path = gcs_folder_path + "/"
        elif not gcs_folder_path:
            self.gcs_folder_path = ""
        else:
            self.gcs_folder_path = gcs_folder_path
        self._build_vector_store()

    def clear_rag_sources(self):
        logging.info("Clearing RAG sources and vector store.")
        self.gcs_bucket_name = None
        self.gcs_folder_path = None
        self.vector_store = None

    def _build_vector_store(self):
        if not self.gcs_bucket_name:
            logging.info("No GCS bucket name set. Vector store will not be built.")
            self.vector_store = None
            return

        logging.info(
            f"Building vector store from GCS: gs://{self.gcs_bucket_name}/{self.gcs_folder_path or ''}..."
        )
        documents = self._get_documents_from_gcs_bucket()
        if not documents:
            logging.warning(
                "No documents could be processed from the GCS location. Vector store not built."
            )
            self.vector_store = None
            return

        self.vector_store = self._create_vector_store_from_documents(documents)
        if self.vector_store:
            logging.info(
                f"Successfully built/updated vector store with {len(documents)} source documents."
            )
        else:
            logging.warning("Failed to build vector store.")

    def _get_documents_from_gcs_bucket(self) -> List[Document]:
        all_documents = []
        if not self.gcs_bucket_name:
            logging.info("No GCS bucket specified for document extraction.")
            return []
        if not self.storage_client:
            logging.error("GCS storage client not initialized. Cannot fetch documents.")
            return []

        try:
            bucket = self.storage_client.bucket(self.gcs_bucket_name)
            blobs = bucket.list_blobs(prefix=self.gcs_folder_path or None)

            found_pdfs = False
            for blob in blobs:
                if blob.name.lower().endswith(".pdf"):
                    if blob.name == self.gcs_folder_path and blob.name.endswith(
                        "/"
                    ):  # Skip "folder" objects
                        continue

                    found_pdfs = True
                    full_gcs_path = f"gs://{self.gcs_bucket_name}/{blob.name}"
                    logging.info(f"Fetching and parsing PDF: {full_gcs_path}")
                    try:
                        pdf_bytes = blob.download_as_bytes()
                        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
                        extracted_text_content = ""
                        for i, page in enumerate(reader.pages):
                            page_text = page.extract_text()
                            if page_text:
                                extracted_text_content += page_text + "\n"

                        if extracted_text_content.strip():
                            all_documents.append(
                                Document(
                                    page_content=extracted_text_content.strip(),
                                    metadata={"source": full_gcs_path},
                                )
                            )
                        else:
                            logging.warning(
                                f"No text extracted from PDF: {full_gcs_path}"
                            )
                    except Exception as pdf_e:
                        logging.error(
                            f"Error processing PDF file {full_gcs_path}: {pdf_e}"
                        )
                        logging.error(traceback.format_exc())
            if not found_pdfs:
                logging.warning(
                    f"No PDF files found in gs://{self.gcs_bucket_name}/{self.gcs_folder_path or ''}"
                )

        except Exception as e:
            logging.error(
                f"Error listing or accessing GCS bucket gs://{self.gcs_bucket_name}/: {e}"
            )
            logging.error(traceback.format_exc())
        return all_documents

    def _create_vector_store_from_documents(
        self, documents: List[Document]
    ) -> Optional[VectorStore]:  # Type hint to base class VectorStore
        if not documents:
            logging.warning("No documents provided to create vector store.")
            return None
        split_documents = self.text_splitter.split_documents(documents)
        if not split_documents:
            logging.warning("Text splitting resulted in no document chunks.")
            return None
        logging.info(
            f"Split {len(documents)} documents into {len(split_documents)} chunks."
        )
        try:
            # InMemoryVectorStore.from_documents is a valid way to create and populate
            vector_store = InMemoryVectorStore.from_documents(
                documents=split_documents,
                embedding=self.embeddings_model,  # Use 'embedding' kwarg
            )
            logging.info("Successfully created in-memory vector store from documents.")
            return vector_store
        except Exception as e:
            logging.error(f"Failed to create vector store: {e}")
            logging.error(traceback.format_exc())
            return None

    def generate_gemini_response(self, prompt: str) -> str:
        prompt_prefix = f"{self.base_prompt}\n\n" if self.base_prompt else ""
        full_prompt = f"{prompt_prefix}User Input: {prompt}"
        if not self.base_prompt:
            full_prompt = prompt
        try:
            message = self.model.invoke([HumanMessage(content=full_prompt)])
            response = message.content.strip()
            if "error" in response.lower() or not response:
                logging.warning(
                    f"Gemini Response: Gemini returned an error or empty message: '{response}'"
                )
                return (
                    self.ERROR_MESSAGE
                    + " This could be due to safety filters, rate limiting, or an issue with the prompt. Please try again."
                )
            logging.info(f"Gemini Response: Generated response: {response}")
            return response
        except Exception as e:
            logging.error(f"Gemini Response: Error generating Gemini response: {e}")
            logging.error(traceback.format_exc())
            return (
                self.ERROR_MESSAGE
                + " This is most likely due to rate limiting or an internal error. Wait 30 seconds and try again."
            )

    def generate_gemini_response_with_rag(self, prompt: str) -> str:
        if not self.vector_store:
            logging.warning(
                "No vector store available for RAG. "
                "Ensure GCS bucket/folder are set and processed via update_rag_sources(). "
                "Falling back to standard generation."
            )
            return self.generate_gemini_response(prompt)
        try:
            # InMemoryVectorStore implements the VectorStore interface which includes as_retriever
            if hasattr(self.vector_store, "as_retriever"):
                retriever = self.vector_store.as_retriever(
                    search_kwargs={"k": self.RAG_TOP_K}
                )
                relevant_chunks = retriever.get_relevant_documents(prompt)
            else:  # Should ideally not be reached if vector_store is a valid VectorStore instance
                logging.error(
                    "Vector store is not valid or does not have an 'as_retriever' method."
                )
                relevant_chunks = []
        except Exception as e:
            logging.error(f"Error retrieving documents from vector store: {e}")
            relevant_chunks = []

        prompt_prefix = f"{self.base_prompt}\n\n" if self.base_prompt else ""
        if relevant_chunks:
            context_str = "\n\n".join([chunk.page_content for chunk in relevant_chunks])
            logging.info(
                f"Retrieved {len(relevant_chunks)} relevant chunks for context."
            )
            full_prompt = (
                f"{prompt_prefix}"
                f"Based on the following information from the provided documents:\n"
                f"--- DOCUMENT CONTEXT START ---\n"
                f"{context_str}\n"
                f"--- DOCUMENT CONTEXT END ---\n\n"
                f"User's question: {prompt}"
            )
        else:
            logging.info(
                "No relevant chunks found from documents for RAG. "
                "Falling back to standard generation."
            )
            return self.generate_gemini_response(prompt)

        try:
            message = self.model.invoke([HumanMessage(content=full_prompt)])
            response = message.content.strip()
            if "error" in response.lower() or not response:
                logging.warning(
                    f"Gemini RAG Response: Gemini returned an error or empty message: '{response}'"
                )
                return (
                    self.ERROR_MESSAGE
                    + " RAG: This could be due to safety filters, rate limiting, or an issue with the prompt and context."
                )
            logging.info(f"Gemini RAG Response: Generated response with RAG context.")
            return response
        except Exception as e:
            logging.error(
                f"Gemini RAG Response: Error generating Gemini response with RAG: {e}"
            )
            logging.error(traceback.format_exc())
            return self.ERROR_MESSAGE + " RAG: Error during final response generation."
