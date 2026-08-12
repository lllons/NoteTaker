"""Optional LLM provider plugins.

The core application does not require an API key. Providers are deliberately
small so local models or another OpenAI-compatible endpoint can be plugged in
without changing extraction, storage, or rendering code.
"""

from __future__ import annotations

import json
import urllib.request
from typing import Any, Protocol


class LLMProvider(Protocol):
    name: str

    def enrich(self, transcript: str, local_facts: dict[str, Any]) -> dict[str, Any]:
        """Return only JSON facts that can be cited back to the transcript."""


EXTRACTION_SYSTEM_PROMPT = """You are a high-fidelity knowledge archivist.
Extract information, do not summarize away useful detail. Never add facts that
are not explicitly supported by the transcript. Return JSON only with arrays
for concepts, definitions, explanations, examples, analogies, formulas,
statistics, code_snippets, action_items, decisions, open_questions, resources,
and inferred_items. Every item must include source_segment_ids when available.
Mark uncertainty instead of guessing. Preserve names, numbers, URLs, commands,
file paths, equations, caveats, assumptions, trade-offs, and exceptions."""


class OpenAICompatibleProvider:
    name = "openai-compatible"

    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model

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
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=90) as response:
            result = json.loads(response.read().decode("utf-8"))
        content = result["choices"][0]["message"]["content"]
        parsed = json.loads(content) if isinstance(content, str) else content
        return parsed if isinstance(parsed, dict) else {}


def provider_from_config(config: Any) -> LLMProvider | None:
    if config.llm_base_url and config.llm_api_key and config.llm_model:
        return OpenAICompatibleProvider(config.llm_base_url, config.llm_api_key, config.llm_model)
    return None
