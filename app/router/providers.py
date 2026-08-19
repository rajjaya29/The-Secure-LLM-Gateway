"""Upstream LLM Provider implementations."""

import time
import uuid
import asyncio
from typing import Dict, Any, Optional
import httpx
from app.config import settings
from app.schemas.openai import ChatCompletionRequest, ChatCompletionResponse, ChatChoice, ChatMessage, UsageInfo


class BaseLLMProvider:
    def __init__(self, name: str):
        self.name = name
        self.total_calls = 0
        self.total_latency_ms = 0.0

    async def generate(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        raise NotImplementedError


class MockLLMProvider(BaseLLMProvider):
    """
    Mock LLM Provider for local development, integration tests, and realistic latency simulation.
    """

    def __init__(
        self,
        name: str = "mock",
        simulated_latency_ms: Optional[float] = None,
        fail_mode: bool = False,
    ):
        super().__init__(name)
        self.simulated_latency_ms = simulated_latency_ms if simulated_latency_ms is not None else settings.MOCK_LLM_LATENCY_MS
        self.fail_mode = fail_mode

    async def generate(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        t0 = time.perf_counter()

        if self.fail_mode:
            raise RuntimeError(f"Mock Provider '{self.name}' simulated upstream failure")

        # Simulate realistic upstream LLM inference/network latency
        if self.simulated_latency_ms > 0:
            await asyncio.sleep(self.simulated_latency_ms / 1000.0)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.total_calls += 1
        self.total_latency_ms += elapsed_ms

        user_content = ""
        for m in reversed(request.messages):
            if m.role == "user":
                user_content = m.content
                break

        # Generate realistic domain answers for benchmark queries
        content_lower = user_content.lower()
        if "capital of france" in content_lower:
            reply = "The capital of France is Paris. It is also known as the City of Light."
        elif "quantum superposition" in content_lower:
            reply = "Quantum superposition is a fundamental principle of quantum mechanics where a system exists in multiple states simultaneously until measured."
        elif "palindrome" in content_lower:
            reply = "def is_palindrome(s: str) -> bool:\n    clean = ''.join(c.lower() for c in s if c.isalnum())\n    return clean == clean[::-1]"
        elif "french revolution" in content_lower:
            reply = "The French Revolution of 1789 was caused by social inequality, economic hardship, heavy taxation, and Enlightenment ideals."
        elif "vector similarity" in content_lower:
            reply = "Vector similarity search uses cosine similarity or Euclidean distance in high-dimensional embedding space to find nearest neighbor documents."
        elif "tcp and udp" in content_lower or "tcp vs udp" in content_lower:
            reply = "TCP is a connection-oriented, reliable protocol with error checking, whereas UDP is connectionless and optimized for low-latency streaming."
        elif "photosynthesis" in content_lower:
            reply = "Photosynthesis converts solar light energy into chemical energy, transforming water and carbon dioxide into glucose and oxygen."
        elif "object-oriented programming" in content_lower or "oop" in content_lower:
            reply = "Core OOP principles include Encapsulation, Abstraction, Inheritance, and Polymorphism."
        elif "transformer" in content_lower:
            reply = "The Transformer architecture relies on self-attention mechanisms, multi-head attention, and positional encoding without recurrent connections."
        elif "supply and demand" in content_lower:
            reply = "The economic law of supply and demand states that market price reaches equilibrium where quantity demanded equals quantity supplied."
        elif "binary search" in content_lower:
            reply = "def binary_search(arr, target):\n    low, high = 0, len(arr) - 1\n    while low <= high:\n        mid = (low + high) // 2\n        if arr[mid] == target: return mid\n        elif arr[mid] < target: low = mid + 1\n        else: high = mid - 1\n    return -1"
        elif "dijkstra" in content_lower:
            reply = "Dijkstra's algorithm finds the shortest path from a starting node to all other nodes in a weighted graph with non-negative edge weights using a priority queue."
        elif "acid" in content_lower:
            reply = "ACID stands for Atomicity, Consistency, Isolation, and Durability, ensuring reliable database transaction processing."
        elif "process" in content_lower and "thread" in content_lower:
            reply = "A process is an independent executing program with its own memory space, while a thread is a lightweight execution unit sharing the process's memory."
        elif "mitochondria" in content_lower:
            reply = "Mitochondria generate cellular energy in the form of ATP through oxidative phosphorylation and the citric acid cycle."
        elif "cryptography" in content_lower or "encryption" in content_lower:
            reply = "Symmetric encryption uses a single shared key for encryption and decryption, whereas asymmetric encryption uses a public/private key pair."
        elif "cap theorem" in content_lower:
            reply = "The CAP theorem states that a distributed data store can simultaneously provide at most two out of three guarantees: Consistency, Availability, and Partition tolerance."
        else:
            reply = f"Processed query successfully via Secure Gateway: '{user_content[:60]}...'. Response generated under {request.model} constraints."

        prompt_tokens = max(1, len(user_content.split()))
        completion_tokens = max(1, len(reply.split()))

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            model=request.model,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=reply),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )


class OpenAIProvider(BaseLLMProvider):
    def __init__(self, api_key: Optional[str], base_url: str = "https://api.openai.com/v1"):
        super().__init__("openai")
        self.api_key = api_key
        self.base_url = base_url

    async def generate(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        if not self.api_key:
            raise ValueError("OpenAI API Key is not configured")
        
        async with httpx.AsyncClient(timeout=settings.PROVIDER_TIMEOUT_SECONDS) as client:
            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
            payload = request.model_dump(exclude_none=True)
            res = await client.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)
            res.raise_for_status()
            self.total_calls += 1
            return ChatCompletionResponse(**res.json())


class AnthropicProvider(BaseLLMProvider):
    def __init__(self, api_key: Optional[str], base_url: str = "https://api.anthropic.com/v1"):
        super().__init__("anthropic")
        self.api_key = api_key
        self.base_url = base_url

    async def generate(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        if not self.api_key:
            raise ValueError("Anthropic API Key is not configured")
        
        system_prompt = ""
        anthropic_messages = []
        for m in request.messages:
            if m.role == "system":
                system_prompt = m.content
            else:
                anthropic_messages.append({"role": m.role, "content": m.content})

        payload = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": anthropic_messages,
            "max_tokens": request.max_tokens or 1024,
        }
        if system_prompt:
            payload["system"] = system_prompt

        async with httpx.AsyncClient(timeout=settings.PROVIDER_TIMEOUT_SECONDS) as client:
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            }
            res = await client.post(f"{self.base_url}/messages", json=payload, headers=headers)
            res.raise_for_status()
            data = res.json()
            self.total_calls += 1
            reply_text = data.get("content", [{}])[0].get("text", "")
            return ChatCompletionResponse(
                id=data.get("id", f"msg-{uuid.uuid4().hex[:8]}"),
                model=request.model,
                choices=[ChatChoice(index=0, message=ChatMessage(role="assistant", content=reply_text), finish_reason="stop")],
                usage=UsageInfo(prompt_tokens=data.get("usage", {}).get("input_tokens", 0), completion_tokens=data.get("usage", {}).get("output_tokens", 0), total_tokens=0),
            )


class OllamaProvider(BaseLLMProvider):
    def __init__(self, base_url: str = "http://localhost:11434", default_model: str = "llama3"):
        super().__init__("ollama")
        self.base_url = base_url
        self.default_model = default_model

    async def generate(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        async with httpx.AsyncClient(timeout=settings.PROVIDER_TIMEOUT_SECONDS) as client:
            payload = {
                "model": request.model if "llama" in request.model else self.default_model,
                "messages": [{"role": m.role, "content": m.content} for m in request.messages],
                "stream": False,
            }
            res = await client.post(f"{self.base_url}/api/chat", json=payload)
            res.raise_for_status()
            data = res.json()
            self.total_calls += 1
            reply_text = data.get("message", {}).get("content", "")
            return ChatCompletionResponse(
                id=f"ollama-{uuid.uuid4().hex[:8]}",
                model=request.model,
                choices=[ChatChoice(index=0, message=ChatMessage(role="assistant", content=reply_text), finish_reason="stop")],
                usage=UsageInfo(prompt_tokens=data.get("prompt_eval_count", 0), completion_tokens=data.get("eval_count", 0), total_tokens=0),
            )
