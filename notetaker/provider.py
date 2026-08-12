"""Optional LLM provider plugins.

The core application does not require an API key. Providers are deliberately
small so local models or another OpenAI-compatible endpoint can be plugged in
without changing extraction, storage, or rendering code.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Protocol


class LLMProvider(Protocol):
    name: str

    def enrich(self, transcript: str, local_facts: dict[str, Any]) -> dict[str, Any]:
        """Return only JSON facts that can be cited back to the transcript."""


EXTRACTION_SYSTEM_PROMPT = """You are a meticulous knowledge archivist, not a summarizer.
Your objective is maximum information retention with zero unsupported claims.
Use only the supplied transcript. Preserve definitions, reasoning chains,
examples, comparisons, exceptions, caveats, formulas, statistics, numbers,
units, names, organizations, products, locations, dates, deadlines, code,
commands, URLs, file paths, APIs, algorithms, and implementation details.
Remove only obvious filler and exact repetition; retain similar statements when
they add context. Return one JSON object with these array keys:
concepts, definitions, explanations, examples, analogies, formulas, statistics,
code_snippets, action_items, decisions, open_questions, resources, inferred_items.
Every object must include source_segment_ids copied from the transcript markers.
If evidence is incomplete, use an uncertainty field or inferred_items; never
complete a fact from general knowledge. Do not rename technical terms. Do not
merge contradictory claims. Keep output concise only where the source repeats
itself, never where it contains a unique detail."""


class OpenAICompatibleProvider:
    name = "openai-compatible"

    def __init__(self, base_url: str, api_key: str, model: str, timeout: int = 45) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def enrich(self, transcript: str, local_facts: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps({"transcript": transcript, "local_facts": local_facts}, ensure_ascii=False),
                },
            ],
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
            content = result["choices"][0]["message"]["content"]
            parsed = json.loads(content) if isinstance(content, str) else content
        except (urllib.error.URLError, TimeoutError, KeyError, IndexError, TypeError, ValueError) as exc:
            raise RuntimeError(f"structured provider response unavailable: {type(exc).__name__}") from exc
        return parsed if isinstance(parsed, dict) else {}


def provider_from_config(config: Any) -> LLMProvider | None:
    if config.llm_base_url and config.llm_api_key and config.llm_model:
        return OpenAICompatibleProvider(config.llm_base_url, config.llm_api_key, config.llm_model, config.provider_timeout)
    return None
