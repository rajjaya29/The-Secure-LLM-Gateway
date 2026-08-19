"""LLM Provider implementations (Mock, OpenAI, Anthropic, Ollama)."""

import time
import uuid
import asyncio
import httpx
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from app.schemas.openai import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
    ChatMessage,
    UsageInfo,
)

logger = logging.getLogger("secure_gateway.providers")


class BaseLLMProvider(ABC):
    def __init__(self, name: str, timeout_seconds: float = 30.0):
        self.name = name
        self.timeout_seconds = timeout_seconds

    @abstractmethod
    async def generate(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        pass


class MockLLMProvider(BaseLLMProvider):
    def __init__(
        self,
        name: str = "mock",
        simulated_latency_ms: float = 80.0,
        fail_mode: bool = False,
    ):
        super().__init__(name=name)
        self.simulated_latency_ms = simulated_latency_ms
        self.fail_mode = fail_mode

    async def generate(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        if self.fail_mode:
            raise RuntimeError(f"Simulated failure from {self.name} provider for testing fallback routing.")

        if self.simulated_latency_ms > 0:
            await asyncio.sleep(self.simulated_latency_ms / 1000.0)

        user_content = ""
        for msg in reversed(request.messages):
            if msg.role == "user":
                user_content = msg.content
                break

        reply_content = self._craft_mock_response(user_content, request.model)
        
        prompt_tokens = max(10, len(user_content.split()) * 2)
        completion_tokens = max(15, len(reply_content.split()) * 2)

        return ChatCompletionResponse(
            id=f"chatcmpl-mock-{uuid.uuid4().hex[:12]}",
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=reply_content),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    def _craft_mock_response(self, prompt: str, model: str) -> str:
        p_lower = prompt.lower()
        if "capital of france" in p_lower:
            return "The capital of France is Paris. It is also known as the City of Light."
        elif "capital of" in p_lower:
            return f"Regarding your inquiry about the capital in '{prompt}': The capital is a major political, economic, and cultural hub."
        elif "python" in p_lower or "code" in p_lower:
            return "Here is a clean Python snippet:\n```python\ndef process_gateway_request(data):\n    return {'status': 'success', 'data': data}\n```"
        elif "weather" in p_lower:
            return "The weather is currently clear and sunny with mild temperatures and gentle breeze."
        elif "who are you" in p_lower or "what are you" in p_lower:
            return f"I am an AI assistant routed through The Secure LLM Gateway using the {model} model."
        else:
            return f"Processed query successfully via Secure Gateway: '{prompt}'. Response generated under {model} model constraints."


class OpenAIProvider(BaseLLMProvider):
    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://api.openai.com/v1", timeout_seconds: float = 30.0):
        super().__init__(name="openai", timeout_seconds=timeout_seconds)
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    async def generate(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        if not self.api_key:
            raise ValueError("OpenAI API key is not configured in gateway settings.")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = request.model_dump(exclude_none=True, exclude={"bypass_cache", "bypass_guardrails"})

        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            resp = await client.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)
            if resp.status_code != 200:
                raise RuntimeError(f"OpenAI API error ({resp.status_code}): {resp.text}")
            return ChatCompletionResponse(**resp.json())


class AnthropicProvider(BaseLLMProvider):
    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://api.anthropic.com/v1", timeout_seconds: float = 30.0):
        super().__init__(name="anthropic", timeout_seconds=timeout_seconds)
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    async def generate(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        if not self.api_key:
            raise ValueError("Anthropic API key is not configured in gateway settings.")

        system_msg = ""
        anthropic_messages = []
        for msg in request.messages:
            if msg.role == "system":
                system_msg += msg.content + "\n"
            else:
                anthropic_messages.append({"role": msg.role, "content": msg.content})

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {
            "model": request.model if "claude" in request.model else "claude-3-5-sonnet-20241022",
            "messages": anthropic_messages,
            "max_tokens": request.max_tokens or 1024,
        }
        if system_msg.strip():
            payload["system"] = system_msg.strip()

        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            resp = await client.post(f"{self.base_url}/messages", json=payload, headers=headers)
            if resp.status_code != 200:
                raise RuntimeError(f"Anthropic API error ({resp.status_code}): {resp.text}")
            
            data = resp.json()
            content_text = "".join([block.get("text", "") for block in data.get("content", [])])
            return ChatCompletionResponse(
                id=f"chatcmpl-ant-{data.get('id', uuid.uuid4().hex[:12])}",
                model=data.get("model", request.model),
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message=ChatMessage(role="assistant", content=content_text),
                        finish_reason="stop",
                    )
                ],
                usage=UsageInfo(
                    prompt_tokens=data.get("usage", {}).get("input_tokens", 0),
                    completion_tokens=data.get("usage", {}).get("output_tokens", 0),
                    total_tokens=data.get("usage", {}).get("input_tokens", 0) + data.get("usage", {}).get("output_tokens", 0),
                ),
            )


class OllamaProvider(BaseLLMProvider):
    def __init__(self, base_url: str = "http://localhost:11434", default_model: str = "llama3", timeout_seconds: float = 60.0):
        super().__init__(name="ollama", timeout_seconds=timeout_seconds)
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model

    async def generate(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        ollama_messages = [{"role": m.role, "content": m.content} for m in request.messages]
        payload = {
            "model": request.model if request.model not in ["gpt-4o", "gpt-4o-mini", "mock"] else self.default_model,
            "messages": ollama_messages,
            "stream": False,
        }

        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            resp = await client.post(f"{self.base_url}/api/chat", json=payload)
            if resp.status_code != 200:
                raise RuntimeError(f"Ollama API error ({resp.status_code}): {resp.text}")
            
            data = resp.json()
            msg = data.get("message", {})
            prompt_eval = data.get("prompt_eval_count", 0)
            eval_count = data.get("eval_count", 0)

            return ChatCompletionResponse(
                id=f"chatcmpl-ollama-{uuid.uuid4().hex[:12]}",
                model=data.get("model", request.model),
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message=ChatMessage(role=msg.get("role", "assistant"), content=msg.get("content", "")),
                        finish_reason="stop",
                    )
                ],
                usage=UsageInfo(
                    prompt_tokens=prompt_eval,
                    completion_tokens=eval_count,
                    total_tokens=prompt_eval + eval_count,
                ),
            )
