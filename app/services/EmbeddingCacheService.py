"""
Service for managing cached embeddings.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class EmbeddingCacheService:
    """Service for managing cached embeddings from embeddings cache directory."""
    
    def __init__(self, cache_directory: str = "data/embeddings_cache"):
        """
        Initialize the EmbeddingCacheService.
        
        Args:
            cache_directory: Directory where embeddings cache files stored
        """
        self.cache_directory = Path(cache_directory)
        self.cache_directory.mkdir(parents=True, exist_ok=True)
        self.embeddings_file = self.cache_directory / "embeddings.json"
        self._cached_embeddings: Optional[Dict[str, List[float]]] = None
        
    def load_embeddings(self) -> Dict[str, List[float]]:
        """
        Load embeddings from the cache file.
        
        Returns:
            Dictionary mapping text to embedding vectors
        """
        if self._cached_embeddings is not None:
            return self._cached_embeddings
            
        try:
            if not self.embeddings_file.exists():
                logger.warning(f"Embeddings file not found at {self.embeddings_file}")
                self._cached_embeddings = {}
                return self._cached_embeddings
            
            with open(self.embeddings_file, 'r', encoding='utf-8') as f:
                self._cached_embeddings = json.load(f)
            
            logger.info(f"Loaded {len(self._cached_embeddings)} cached embeddings")
            return self._cached_embeddings
            
        except Exception as e:
            logger.error(f"Error loading embeddings from cache: {e}")
            self._cached_embeddings = {}
            return self._cached_embeddings
    
    def save_embeddings(self, embeddings: Dict[str, List[float]]) -> bool:
        """
        Save embeddings to the cache file.
        
        Args:
            embeddings: Dictionary mapping text to embedding vectors
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            with open(self.embeddings_file, 'w', encoding='utf-8') as f:
                json.dump(embeddings, f, indent=2)
            
            # Update cached embeddings
            self._cached_embeddings = embeddings
            logger.info(f"Saved {len(embeddings)} embeddings to cache")
            return True
            
        except Exception as e:
            logger.error(f"Error saving embeddings to cache: {e}")
            return False
    
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """
        Get embedding for a specific text.
        
        Args:
            text: Text to get embedding for
            
        Returns:
            Embedding vector if found, None otherwise
        """
        embeddings = self.load_embeddings()
        return embeddings.get(text)
    
    def has_embedding(self, text: str) -> bool:
        """
        Check if embedding exists for a text.
        
        Args:
            text: Text to check
            
        Returns:
            True if embedding exists, False otherwise
        """
        embeddings = self.load_embeddings()
        return text in embeddings
    
    def add_embedding(self, text: str, embedding: List[float]) -> bool:
        """
        Add a single embedding to the cache.
        
        Args:
            text: Text content
            embedding: Embedding vector
            
        Returns:
            True if added successfully, False otherwise
        """
        embeddings = self.load_embeddings()
        embeddings[text] = embedding
        return self.save_embeddings(embeddings)
    
    def get_cache_info(self) -> Dict[str, any]:
        """
        Get information about the embeddings cache.
        
        Returns:
            Dictionary with cache information
        """
        embeddings = self.load_embeddings()
        file_size = self.embeddings_file.stat().st_size if self.embeddings_file.exists() else 0
        
        return {
            'cache_directory': str(self.cache_directory),
            'embeddings_file': str(self.embeddings_file),
            'total_embeddings': len(embeddings),
            'file_size_bytes': file_size,
            'file_exists': self.embeddings_file.exists()
        }
    
    def clear_cache(self) -> bool:
        """
        Clear the embeddings cache.
        
        Returns:
            True if cleared successfully, False otherwise
        """
        try:
            if self.embeddings_file.exists():
                self.embeddings_file.unlink()
            self._cached_embeddings = {}
            logger.info("Cleared embeddings cache")
            return True
        except Exception as e:
            logger.error(f"Error clearing embeddings cache: {e}")
            return False 