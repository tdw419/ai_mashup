# engine/llm.py
import os, json, requests
from typing import Callable

class LLM:
    """
    Minimal OpenAI-compatible client.
    Defaults to LM Studio at http://localhost:1234/v1 with model from LLM_MODEL.
    Set:
      LLM_BASE_URL (default http://localhost:1234/v1)
      LLM_API_KEY  (optional; not required for LM Studio)
      LLM_MODEL    (default 'qwen2.5-coder:7b-instruct' or any local)
    """
    def __init__(self):
        self.base = os.getenv("LLM_BASE_URL", "http://localhost:1234/v1")
        self.key  = os.getenv("LLM_API_KEY", "")
        self.model= os.getenv("LLM_MODEL", "qwen2.5-coder:7b-instruct")
        self.session = requests.Session()

    def __call__(self, prompt: str) -> str:
        url = f"{self.base}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        body = {
            "model": self.model,
            "messages": [{"role":"system","content":"You are a decisive senior systems builder."},
                         {"role":"user","content":prompt}],
            "temperature": 0.3
        }
        r = self.session.post(url, headers=headers, data=json.dumps(body), timeout=120)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()
