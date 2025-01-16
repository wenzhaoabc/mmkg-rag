import os
from openai import (
    AsyncOpenAI,
)
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionSystemMessageParam,
)


class LLM:
    _client_async: AsyncOpenAI | None = None

    @staticmethod
    def _get_instance(
        base_url: str | None = None, api_key: str | None = None
    ) -> AsyncOpenAI:
        if LLM._client_async is None:
            base_url = base_url or os.environ.get("BASE_URL")
            api_key = api_key or os.environ.get("API_KEY")
            llm_model = os.environ.get("LLM_MODEL") or "gpt-4o-mini"
            assert base_url, "base_url is required"
            assert api_key, "api_key is required"
            LLM._client_async = AsyncOpenAI(base_url=base_url, api_key=api_key)
        return LLM._client_async

    @staticmethod
    async def chat(
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list[dict] = [],
        model: str = "gpt-4o",
        base_url: str | None = None,
        api_key: str | None = None,
        **kwargs,
    ) -> str:
        client_async = LLM._get_instance(base_url=base_url, api_key=api_key)
        model = os.environ.get("LLM_MODEL") or "gpt-4o-mini"

        messages: list[ChatCompletionMessageParam] = []
        if system_prompt:
            messages.append(
                ChatCompletionSystemMessageParam(role="system", content=system_prompt)
            )
        if history_messages:
            messages.extend(history_messages)
        messages.append(ChatCompletionUserMessageParam(role="user", content=prompt))

        response: ChatCompletion = await client_async.chat.completions.create(
            model=model, messages=messages, max_tokens=LLM.max_tokens(model), **kwargs
        )
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("Response content is None")
        return content

    @staticmethod
    async def chat_msg_sync(
        messages: list[dict],
        model: str = "gpt-4o",
        base_url: str | None = None,
        api_key: str | None = None,
        **kwargs,
    ) -> str:
        client_async = LLM._get_instance(base_url=base_url, api_key=api_key)
        model = os.environ.get("LLM_MODEL") or "gpt-4o-mini"
        
        response: ChatCompletion = await client_async.chat.completions.create(
            model=model, messages=messages, max_tokens=LLM.max_tokens(model), **kwargs
        )
        return response.choices[0].message.content

    @staticmethod
    def max_tokens(model: str = "glm-4-plus") -> int:
        max_tokens = {
            "gpt-4o-mini": 8192,
            "gpt-4o": 8192,
            "claude-3.5-sonnet": 8192,
            "glm-4-flashx": 8192,
            "glm-4-plus": 8192,
        }
        if model not in max_tokens:
            raise ValueError(f"Model {model} not supported")
        return max_tokens[model]
