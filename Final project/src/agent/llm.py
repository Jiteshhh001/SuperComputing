"""
═══════════════════════════════════════════════════════════════════════════════
  LLM Client — OpenRouter chat completions with model fallback
  Free-tier models are frequently rate-limited or retired, so every call tries
  a short list of models in order and remembers the last one that worked.
═══════════════════════════════════════════════════════════════════════════════
"""

import logging
import time
from typing import Optional

import httpx

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import (
    OPENROUTER_BASE_URL, LLM_MODEL, LLM_FALLBACK_MODELS,
    LLM_TEMPERATURE, LLM_MAX_TOKENS, LLM_TIMEOUT_SECONDS, get_openrouter_api_key,
)

logger = logging.getLogger(__name__)


class LLMClient:
    """Small OpenRouter client used by the agent and the recommender."""

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key
        self.models = list(dict.fromkeys([LLM_MODEL, *LLM_FALLBACK_MODELS]))
        self.last_model: Optional[str] = None
        self.last_error: Optional[str] = None
        self._cooldown: dict[str, float] = {}   # model -> time until which it is skipped

    @property
    def api_key(self) -> str:
        # Read lazily so a key added in Settings / Streamlit secrets is picked up
        return self._api_key or get_openrouter_api_key()

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def set_api_key(self, key: str):
        self._api_key = key.strip() or None

    def complete(
        self,
        messages: list[dict],
        max_tokens: int = LLM_MAX_TOKENS,
        temperature: float = LLM_TEMPERATURE,
        json_mode: bool = False,
        time_budget: float = 45.0,
    ) -> Optional[str]:
        """
        Return the assistant text, or None if no model produced an answer
        within the time budget. Never raises.
        """
        if not self.api_key:
            self.last_error = "No OpenRouter API key configured"
            return None

        start = time.time()
        ordered = ([self.last_model] if self.last_model else []) + self.models
        for model in dict.fromkeys(ordered):
            if self._cooldown.get(model, 0) > time.time():
                continue
            remaining = time_budget - (time.time() - start)
            if remaining < 5:
                break
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "reasoning": {"effort": "low", "exclude": True},
            }
            if json_mode:
                payload["response_format"] = {"type": "json_object"}
            try:
                response = httpx.post(
                    f"{OPENROUTER_BASE_URL}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://github.com/Jiteshhh001/SuperComputing",
                        "X-Title": "LearnFlow AI",
                    },
                    json=payload,
                    timeout=min(LLM_TIMEOUT_SECONDS, remaining),
                )
                if response.status_code == 200:
                    choice = (response.json().get("choices") or [{}])[0]
                    content = ((choice.get("message") or {}).get("content") or "").strip()
                    if content:
                        self.last_model = model
                        self.last_error = None
                        return content
                    self.last_error = f"{model}: empty response"
                else:
                    self.last_error = f"{model}: HTTP {response.status_code}"
                    if response.status_code in (401, 402):
                        break               # bad key / no credit — other models won't help
                    if response.status_code in (404, 429):
                        self._cooldown[model] = time.time() + (600 if response.status_code == 404 else 60)
            except Exception as e:  # network errors, timeouts
                self.last_error = f"{model}: {type(e).__name__}"
            logger.warning(f"LLM call failed — {self.last_error}")
        return None
