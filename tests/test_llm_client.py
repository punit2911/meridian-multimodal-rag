import pytest

from meridian.config import LLMSettings
from meridian.llm.base import LLMError
from meridian.llm.openai_client import OpenAIClient


def test_docker_host_ollama_does_not_require_api_key():
    client = OpenAIClient(LLMSettings(
        provider="openai",
        model="llama3.1",
        openai_base_url="http://host.docker.internal:11434/v1",
    ))

    assert client.url == "http://host.docker.internal:11434/v1/chat/completions"


def test_nonlocal_openai_compatible_url_still_requires_api_key():
    with pytest.raises(LLMError, match="OPENAI_API_KEY is not set"):
        OpenAIClient(LLMSettings(
            provider="openai",
            openai_base_url="http://host.docker.internal.example.com/v1",
        ))
