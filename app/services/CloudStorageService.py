import os
import logging
from datetime import datetime, timedelta
from typing import Optional, Tuple

from google.cloud import storage
from google.oauth2 import service_account # Used if explicit credentials path is given

# Configure logging for the application
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./credentials.json"

class CloudStorageService:
    """
    A class to facilitate uploading local files to Google Cloud Storage (GCS)
    and generating signed URLs for downloading those files.
    """
    def __init__(self, bucket_name: str = "ashes_project_website_artifacts", credentials_path: Optional[str] = "./credentials.json"):
        """
        Initializes the CloudStorageService with the target bucket and optional credentials.

        Args:
            bucket_name (str): The name of the GCS bucket to interact with. (REQUIRED)
            credentials_path (Optional[str]): Path to a Google Cloud service account key file (JSON).
                                              If None, the client will attempt to authenticate
                                              using default credentials (e.g., GOOGLE_APPLICATION_CREDENTIALS
                                              environment variable, GCE metadata service). Defaults to None.
        Raises:
            RuntimeError: If GCS client fails to initialize or connect to the bucket.
            FileNotFoundError: If the specified credentials_path does not exist.
        """
        self.bucket_name = bucket_name
        self.client: Optional[storage.Client] = None
        self.bucket: Optional[storage.Bucket] = None

        try:
            if credentials_path:
                if not os.path.exists(credentials_path):
                    raise FileNotFoundError(f"Credentials file not found at: {credentials_path}")
                credentials = service_account.Credentials.from_service_account_file(credentials_path)
                self.client = storage.Client(credentials=credentials)
                logger.info(f"GCS client initialized using credentials from: {credentials_path}")
            else:
                self.client = storage.Client()
                logger.info("GCS client initialized using default credentials (e.g., GOOGLE_APPLICATION_CREDENTIALS).")

            # Create bucket reference without checking existence
            self.bucket = self.client.bucket(bucket_name)
            logger.info(f"Using GCS bucket: {bucket_name}")
            
            # Skip the existence check to avoid permission issues
            # The actual permissions will be checked during operations
            pass

        except Exception as e:
            logger.error(f"Failed to initialize CloudStorageService for bucket '{bucket_name}': {e}")
            raise RuntimeError(f"CloudStorageService initialization failed: {e}")

    def upload_file(self, local_file_path: str, destination_blob_name: str) -> storage.blob.Blob:
        """
        Uploads a local file to the specified destination in the GCS bucket.

        Args:
            local_file_path (str): The path to the local file to upload.
            destination_blob_name (str): The name of the file (blob) in the GCS bucket.
                                         This can include subdirectories (e.g., "audio/my_file.wav").

        Returns:
            google.cloud.storage.blob.Blob: The Blob object representing the uploaded file.

        Raises:
            FileNotFoundError: If the local file does not exist.
            RuntimeError: If the GCS bucket is not initialized.
            Exception: For any GCS upload errors.
        """
        if not os.path.exists(local_file_path):
            raise FileNotFoundError(f"Local file not found: {local_file_path}")

        if not self.bucket:
            raise RuntimeError("GCS bucket not initialized. Call __init__ first.")

        blob = self.bucket.blob(destination_blob_name)
        try:
            logger.info(f"Attempting to upload '{local_file_path}' to '{destination_blob_name}' in bucket '{self.bucket_name}'...")
            blob.upload_from_filename(local_file_path)
            logger.info(f"File '{local_file_path}' successfully uploaded to '{destination_blob_name}'.")
            return blob
        except Exception as e:
            logger.error(f"Failed to upload '{local_file_path}' to '{destination_blob_name}': {e}")
            raise

    def get_signed_url(self, blob_name: str, expiration_seconds: int = 3600) -> str:
        """
        Generates a signed URL for a specific blob (file) in the GCS bucket.
        This URL provides temporary access to the file without requiring Google credentials.

        Args:
            blob_name (str): The name of the blob (file) in the GCS bucket.
            expiration_seconds (int): The duration in seconds for which the signed URL will be valid.
                                      Defaults to 3600 seconds (1 hour).

        Returns:
            str: The generated signed URL.

        Raises:
            FileNotFoundError: If the specified blob does not exist in the bucket.
            RuntimeError: If the GCS bucket is not initialized.
            Exception: For any GCS signed URL generation errors.
        """
        if not self.bucket:
            raise RuntimeError("GCS bucket not initialized. Call __init__ first.")

        blob = self.bucket.blob(blob_name)
        # Ensure the blob exists before attempting to generate a signed URL
        if not blob.exists():
            raise FileNotFoundError(f"Blob '{blob_name}' not found in bucket '{self.bucket_name}'. Cannot generate signed URL.")

        try:
            # generate_signed_url expects a datetime object for expiration (absolute time)
            expiration_time = datetime.now() + timedelta(seconds=expiration_seconds)

            signed_url = blob.generate_signed_url(
                version="v4",
                expiration=expiration_time,
                method="GET"  # Specifies that this URL is for downloading (GET request)
            )
            logger.info(f"Generated signed URL for '{blob_name}', valid for {expiration_seconds} seconds.")
            return signed_url
        except Exception as e:
            logger.error(f"Failed to generate signed URL for '{blob_name}': {e}")
            raise

    def upload_and_get_signed_url(
        self,
        local_file_path: str,
        destination_blob_name: str,
        expiration_seconds: int = 3600
    ) -> str:
        """
        Combines the upload and signed URL generation steps.

        Uploads a local file to GCS and then generates a signed URL for that uploaded file.

        Args:
            local_file_path (str): The path to the local file to upload.
            destination_blob_name (str): The name of the file (blob) in the GCS bucket.
            expiration_seconds (int): The duration in seconds for which the signed URL will be valid.
                                      Defaults to 3600 seconds (1 hour).

        Returns:
            str: The generated signed URL for the uploaded file.

        Raises:
            FileNotFoundError: If the local file does not exist.
            Exception: For any GCS upload or signed URL generation errors.
        """
        # First, upload the file
        self.upload_file(local_file_path, destination_blob_name)

        # Then, get the signed URL for the uploaded file
        signed_url = self.get_signed_url(destination_blob_name, expiration_seconds)

        return signed_url