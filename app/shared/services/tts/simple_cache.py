"""Simple in-memory LRU cache for TTS results."""

import hashlib
import json
import time
import threading
import logging
from typing import Dict, Any, Optional, Tuple
from functools import wraps

logger = logging.getLogger(__name__)


class SimpleCache:
    """
    Simple in-memory LRU cache for TTS results.
    
    Note: This is per-worker. With multiple workers, cache is not shared.
    This is acceptable - each worker maintains its own cache.
    """
    
    def __init__(self, max_size: int = 100, ttl: int = 86400):
        """
        Initialize cache.
        
        Args:
            max_size: Maximum number of cached items
            ttl: Time to live in seconds (default 24h)
        """
        self.cache: Dict[str, Tuple[Any, float]] = {}  # key -> (value, expiry_time)
        self.max_size = max_size
        self.ttl = ttl
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0
    
    def _make_key(self, text: str, **params) -> str:
        """
        Generate cache key from parameters.
        
        Args:
            text: Text to synthesize
            **params: Additional parameters affecting output
            
        Returns:
            Cache key (16 char hex string)
        """
        # Only include params that affect audio output
        cache_params = {k: v for k, v in params.items() if v is not None}
        cache_data = {'text': text, **cache_params}
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.sha256(cache_str.encode()).hexdigest()[:16]
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        with self.lock:
            if key in self.cache:
                value, expiry = self.cache[key]
                
                # Check if expired
                if time.time() < expiry:
                    self.hits += 1
                    return value
                else:
                    # Remove expired entry
                    del self.cache[key]
            
            self.misses += 1
            return None
    
    def set(self, key: str, value: Any) -> None:
        """
        Set in cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        with self.lock:
            # Evict oldest if full (simple FIFO)
            if len(self.cache) >= self.max_size and key not in self.cache:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            
            # Add new entry
            expiry = time.time() + self.ttl
            self.cache[key] = (value, expiry)
    
    def clear_expired(self) -> int:
        """
        Clear expired entries.
        
        Returns:
            Number of entries cleared
        """
        with self.lock:
            now = time.time()
            expired_keys = [
                k for k, (_, exp) in self.cache.items()
                if now >= exp
            ]
            for key in expired_keys:
                del self.cache[key]
            
            if expired_keys:
                logger.debug(f"Cleared {len(expired_keys)} expired cache entries")
            
            return len(expired_keys)
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dict with cache statistics
        """
        with self.lock:
            total = self.hits + self.misses
            hit_rate = (self.hits / total * 100) if total > 0 else 0
            
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': f"{hit_rate:.1f}%"
            }


def cached_tts(cache_instance: SimpleCache):
    """
    Decorator to cache TTS results.
    
    Args:
        cache_instance: SimpleCache instance to use
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, text: str, **kwargs):
            # Generate cache key
            cache_key = cache_instance._make_key(
                text,
                provider=self.provider_name,
                **{k: v for k, v in kwargs.items() if k not in ['output_filepath', 'cloud_storage_service_override']}
            )
            
            # Try cache (only if uploading to cloud)
            if kwargs.get('upload_to_cloud', True):
                cached = cache_instance.get(cache_key)
                if cached:
                    logger.info(
                        f"[{self.provider_name.capitalize()} TTS] Cache hit "
                        f"(stats: {cache_instance.get_stats()})"
                    )
                    return cached
            
            # Generate
            result = func(self, text, **kwargs)
            
            # Cache result (only if cloud URL)
            if result.get('cloud_url'):
                cache_instance.set(cache_key, result)
                logger.debug(f"[{self.provider_name.capitalize()} TTS] Result cached")
            
            return result
        
        return wrapper
    return decorator


# Module-level cache (shared across instances in worker)
_tts_cache = SimpleCache(max_size=100, ttl=86400)

