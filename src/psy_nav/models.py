"""LLM 客户端，兼容 OpenAI 接口（DeepSeek / 本地 vLLM）。"""
from __future__ import annotations

import json
import os
import re

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential


class LLMClient:
    def __init__(self) -> None:
        self._client = AsyncOpenAI(
            api_key=os.getenv("LLM_API_KEY", ""),
            base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1"),
        )
        self._model = os.getenv("LLM_MODEL", "deepseek-chat")
        self._timeout = float(os.getenv("LLM_TIMEOUT", "40"))

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    async def chat(
        self,
        system: str,
        user: str,
        max_tokens: int = 1200,
        temperature: float = 0.3,
        json_mode: bool = True,
    ) -> dict | str:
        kwargs: dict = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "timeout": self._timeout,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        resp = await self._client.chat.completions.create(**kwargs)
        content = resp.choices[0].message.content or ""

        if json_mode:
            return self._parse_json(content)
        return content

    @staticmethod
    def _parse_json(text: str) -> dict:
        text = text.strip()
        # 去除可能的 markdown 代码块
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # 尝试提取第一个 JSON 对象
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group())
                except json.JSONDecodeError:
                    pass
        return {}
