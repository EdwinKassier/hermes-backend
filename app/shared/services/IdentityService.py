import hashlib
from typing import Any, Dict


class IdentityService:
    """
    Service for generating consistent user identifiers based on request information.
    Provides methods to create consistent but anonymized user identifiers.
    """

    @staticmethod
    def generate_user_id(request) -> str:
        """
        Generate a consistent user ID based on multiple factors from the request.

        Args:
            request: The request object containing information about the user.

        Returns:
            str: A hash-based identifier that is consistent but anonymized.
        """
        # Combine multiple identification factors
        identification_factors = [
            request.remote_addr or "",  # IP Address
            request.headers.get("User-Agent", ""),  # Browser/Device info
            request.headers.get("Accept-Language", ""),  # Language preferences
        ]

        # Create a hash-based identifier that's consistent but anonymized
        combined_factors = "|".join(identification_factors)
        user_id = hashlib.sha256(combined_factors.encode()).hexdigest()

        return user_id

    @classmethod
    def get_identity_fingerprint(cls, request) -> Dict[str, Any]:
        """
        Get a dictionary of identity-related information from the request.

        Args:
            request: The request object

        Returns:
            Dict containing various identity-related information
        """
        return {
            "ip_address": request.remote_addr or "",
            "user_agent": request.headers.get("User-Agent", ""),
            "accept_language": request.headers.get("Accept-Language", ""),
            "user_id": cls.generate_user_id(request),
        }
