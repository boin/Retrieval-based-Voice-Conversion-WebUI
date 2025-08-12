import time
import threading
import logging
from typing import Dict, Optional, Callable, Any

from configs.config import Config

logger = logging.getLogger(__name__)


class LRURegistry:
    def __init__(self, config: Config, max_models: int = 2, ttl_seconds: int = 1800, hubert_provider: Optional[Callable[[Config], Any]] = None):
        self.config = config
        self.max_models = max_models
        self.ttl_seconds = ttl_seconds
        self.hubert_provider = hubert_provider
        self._lock = threading.Lock()
        self._map: Dict[str, object] = {}
        self._last_used: Dict[str, float] = {}

    def get(self, model_ckpt: str):
        now = time.time()
        with self._lock:
            # hit
            if model_ckpt in self._map:
                self._last_used[model_ckpt] = now
                return self._map[model_ckpt]
            # create new session
            # lazy import to avoid importing heavy deps when not needed (e.g., in tests)
            from .session import ModelSession  # type: ignore
            session = ModelSession(model_ckpt, self.config, hubert_provider=self.hubert_provider)
            self._map[model_ckpt] = session
            self._last_used[model_ckpt] = now
            self._evict_if_needed(now)
            return session

    def _evict_if_needed(self, now: float):
        # ttl eviction first
        keys_to_delete = [k for k, t in self._last_used.items() if now - t > self.ttl_seconds]
        for k in keys_to_delete:
            self._evict_key(k, reason="ttl")
        # lru eviction
        while len(self._map) > self.max_models:
            # find lru key
            lru_key = min(self._last_used.items(), key=lambda kv: kv[1])[0]
            self._evict_key(lru_key, reason="lru")

    def _evict_key(self, key: str, reason: str):
        try:
            session = self._map.pop(key, None)
            self._last_used.pop(key, None)
            if session:
                logger.info(f"[LRURegistry] Evicting model {key} due to {reason}")
                session.unload()
        except Exception:
            logger.exception("[LRURegistry] Eviction error")

    def clear(self):
        with self._lock:
            keys = list(self._map.keys())
            for k in keys:
                self._evict_key(k, reason="manual-clear")
            
