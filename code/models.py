"""
Unified LLM model classes with a consistent call() interface.
Supports Gemini, GPT, Qwen, and Grok. Credentials are loaded from
the project's credentials/ directory.
"""

import json
from abc import ABC, abstractmethod
from pathlib import Path

# Repo root is two levels above this file: code/ → Policy-Agent/
REPO_ROOT       = Path(__file__).parent.parent
CREDENTIALS_DIR = REPO_ROOT / "credentials"


def _load_api_key(filename: str) -> str:
    """Read 'aki_key' from a JSON credentials file."""
    path = CREDENTIALS_DIR / filename
    with open(path, encoding="utf-8") as f:
        key = json.load(f).get("aki_key", "")
    if not key:
        raise ValueError(f"API key not found in {filename}")
    return key


class BaseModel(ABC):
    """Abstract base for all LLM models."""

    def __init__(self, temperature: float = 0.7, max_tokens: int = 256):
        self.temperature = temperature
        self.max_tokens  = max_tokens
        self.client      = None
        self.model_name  = None
        self._initialize_client()

    @abstractmethod
    def _initialize_client(self): ...

    @abstractmethod
    def call(self, prompt: str) -> str | None:
        """Send prompt; return response text or None on failure."""
        ...

    def get_model_name(self) -> str:
        return self.model_name


class GeminiModel(BaseModel):
    """Google Gemini via google-genai SDK."""

    def __init__(self,
                 temperature: float = 0.7,
                 max_tokens: int = 1024,
                 model_name: str = "gemini-2.5-flash",
                 enable_thinking: bool = False):
        self._model_name_config = model_name
        self.enable_thinking = enable_thinking
        super().__init__(temperature, max_tokens)

    def _initialize_client(self):
        from google import genai
        self.client     = genai.Client(api_key=_load_api_key("gemini_api_key.json"))
        self.model_name = self._model_name_config

    def call(self, prompt: str) -> str | None:
        from google.genai import types
        try:
            thinking = types.ThinkingConfig(thinking_budget=0 if not self.enable_thinking else None)
            resp = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                    response_mime_type="application/json",
                    thinking_config=thinking,
                ),
            )
            return resp.text
        except Exception as e:
            print(f"[Gemini] Error: {e}")
            return None


class GPTModel(BaseModel):
    """OpenAI GPT via openai SDK."""

    def __init__(self,
                 temperature: float = 0.7,
                 max_tokens: int = 256,
                 model_name: str = "gpt-4o-mini"):
        self._model_name_config = model_name
        super().__init__(temperature, max_tokens)

    def _initialize_client(self):
        from openai import OpenAI
        self.client     = OpenAI(api_key=_load_api_key("openai_api_key.json"))
        self.model_name = self._model_name_config

    def call(self, prompt: str) -> str | None:
        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return resp.choices[0].message.content
        except Exception as e:
            print(f"[GPT] Error: {e}")
            return None


class QwenModel(BaseModel):
    """Qwen via SiliconFlow (OpenAI-compatible API)."""

    def __init__(self,
                 temperature: float = 0.7,
                 max_tokens: int = 256,
                 model_name: str = "Qwen/Qwen2.5-7B-Instruct",
                 base_url: str = "https://api.siliconflow.cn/v1"):
        self._model_name_config = model_name
        self.base_url = base_url
        super().__init__(temperature, max_tokens)

    def _initialize_client(self):
        from openai import OpenAI
        self.client     = OpenAI(api_key=_load_api_key("api_key.json"), base_url=self.base_url)
        self.model_name = self._model_name_config

    def call(self, prompt: str) -> str | None:
        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return resp.choices[0].message.content
        except Exception as e:
            print(f"[Qwen] Error: {e}")
            return None


class GrokModel(BaseModel):
    """xAI Grok via OpenAI-compatible API (3600s timeout for reasoning models)."""

    def __init__(self,
                 temperature: float = 0.7,
                 max_tokens: int = 256,
                 model_name: str = "grok-4-1-fast-reasoning",
                 base_url: str = "https://api.x.ai/v1",
                 timeout: float = 3600.0):
        self._model_name_config = model_name
        self.base_url = base_url
        self.timeout  = timeout
        super().__init__(temperature, max_tokens)

    def _initialize_client(self):
        import httpx
        from openai import OpenAI
        self.client = OpenAI(
            api_key=_load_api_key("grok_api_key.json"),
            base_url=self.base_url,
            timeout=httpx.Timeout(self.timeout),
        )
        self.model_name = self._model_name_config

    def call(self, prompt: str) -> str | None:
        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return resp.choices[0].message.content
        except Exception as e:
            print(f"[Grok] Error: {e}")
            return None


# Factory

_MODEL_MAP = {
    "gemini": GeminiModel,
    "gpt":    GPTModel,
    "openai": GPTModel,
    "qwen":   QwenModel,
    "grok":   GrokModel,
}


def create_model(model_type: str, **kwargs) -> BaseModel:
    """Instantiate a model by type name ('gemini', 'gpt', 'qwen', 'grok')."""
    cls = _MODEL_MAP.get(model_type.lower())
    if cls is None:
        raise ValueError(f"Unknown model type '{model_type}'. Choose from: {list(_MODEL_MAP)}")
    return cls(**kwargs)
