import uuid, time, json
from typing import List, Dict, Any, Iterable
from datetime import datetime
from .types import KNode, KState

class MemoryStore:
    """Simple file-backed store; swap with LanceDB later if you want."""
    def __init__(self, path: str = ".cache/qa_ledger.jsonl"):
        self.path = path

    def add(self, nodes: Iterable[KNode]):
        import os
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            for n in nodes:
                f.write(json.dumps(n.to_dict(), default=str) + "\n")

    def all(self) -> List[Dict[str, Any]]:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return [json.loads(x) for x in f]
        except FileNotFoundError:
            return []

def new_question(content: str, deps=None, quality=0.0) -> KNode:
    return KNode(
        id=str(uuid.uuid4()), kind="question", content=content,
        state=KState.RAW_QUESTION, confidence=0.1, deps=deps or [],
        evidence=[], created_at=datetime.utcnow(), updated_at=datetime.utcnow(),
        quality=quality
    )

def new_answer(content: str, deps=None, confidence=0.5, evidence=None, quality=0.0) -> KNode:
    return KNode(
        id=str(uuid.uuid4()), kind="answer", content=content,
        state=KState.PARTIALLY_ANSWERED, confidence=confidence, deps=deps or [],
        evidence=evidence or [], created_at=datetime.utcnow(), updated_at=datetime.utcnow(),
        quality=quality
    )
