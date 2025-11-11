import json
from typing import Dict, Any
import requests
from db_operations import get_db_connection

class LanceDBCurator:
    def __init__(self, db, lm_client_url="http://127.0.0.1:1234/v1/chat/completions"):
        self.db = db
        self.lm_client_url = lm_client_url

    def build_prompt(self, content: str, metadata: Dict[str, Any]) -> str:
        return f"""You are an expert AI Knowledge Curator and Database Organizer. Your primary goal is to analyze documents from a LanceDB vector store, summarize their core content, and suggest improved, highly descriptive metadata tags. This will make the documents more discoverable, relevant, and useful for other AI agents and future LLM chains.

You will receive a document's content and its current metadata. Your task is to:
1. Summarize.
2. Suggest metadata.
3. Detect pxOS primitives.

---DOCUMENT_START---
Content:
{content}

Metadata:
{json.dumps(metadata, indent=2)}
---DOCUMENT_END---

Return ONLY valid JSON with this shape:

{{
    "summary": "...",
    "suggested_metadata": {{
        "stage": "...",
        "category": "...",
        "keywords": ["..."],
        "purpose": "...",
        "contains_pxos_primitives": false,
        "original_model_used": "{metadata.get('model_used', '')}"
    }},
    "justification": "..."
}}
"""

    def curate_doc(self, doc_id: str) -> Dict[str, Any]:
        table = self.db.open_table("documents")
        doc = table.search(f"id = '{doc_id}'").limit(1).to_pandas().to_dict("records")[0]
        if not doc:
            raise ValueError(f"Document {doc_id} not found")

        prompt = self.build_prompt(doc["content"], doc)

        payload = {
            "model": "llama-3.1-8b-instruct",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 1500
        }

        try:
            r = requests.post(self.lm_client_url, json=payload, timeout=300)
            r.raise_for_status()
            raw = r.json()["choices"][0]["message"]["content"]
            data = json.loads(raw)
        except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
            print(f"Error during LLM curation: {e}")
            data = {
                "summary": "",
                "suggested_metadata": {},
                "justification": "LLM returned an error or non-JSON, manual review needed."
            }

        update_fields = {}
        if "summary" in data and data["summary"]:
            update_fields["summary"] = data["summary"]
        if "suggested_metadata" in data and isinstance(data["suggested_metadata"], dict):
            for k, v in data["suggested_metadata"].items():
                if v is not None:
                    update_fields[k] = v

        if update_fields:
            table.update(where=f"id = '{doc_id}'", values=update_fields)

        return data
