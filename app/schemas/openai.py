"""OpenAI API compatible schemas for chat completions and model listings."""

from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel, Field
import time
import uuid


class ChatMessage(BaseModel):
    role: str = Field(..., description="The role of the author of this message: system, user, assistant, function.")
    content: str = Field(..., description="The contents of the message.")
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str = Field(default="gpt-4o-mini", description="ID of the model to use.")
    messages: List[ChatMessage] = Field(..., min_length=1, description="List of messages comprising the conversation.")
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    n: Optional[int] = Field(default=1)
    stream: Optional[bool] = Field(default=False)
    max_tokens: Optional[int] = Field(default=None)
    presence_penalty: Optional[float] = Field(default=0.0)
    frequency_penalty: Optional[float] = Field(default=0.0)
    user: Optional[str] = None
    
    # Gateway-specific bypass / overrides (optional)
    bypass_cache: Optional[bool] = False
    bypass_guardrails: Optional[bool] = False


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: Optional[str] = "stop"


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChoice]
    usage: UsageInfo
    system_fingerprint: Optional[str] = "fp_secure_gateway_v1"


class ModelPermission(BaseModel):
    id: str = Field(default_factory=lambda: f"modelperm-{uuid.uuid4().hex[:12]}")
    object: str = "model_permission"
    created: int = Field(default_factory=lambda: int(time.time()))
    allow_create_engine: bool = False
    allow_sampling: bool = True
    allow_logprobs: bool = True
    allow_search_indices: bool = False
    allow_view: bool = True
    allow_fine_tuning: bool = False
    organization: str = "*"
    group: Optional[str] = None
    is_blocking: bool = False


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "secure-llm-gateway"
    permission: List[ModelPermission] = []
    root: Optional[str] = None
    parent: Optional[str] = None


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard]
