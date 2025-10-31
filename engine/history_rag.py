from dataclasses import dataclass
from datetime import datetime, timezone
import math

@dataclass
class Hit:
    text: str
    ts: datetime
    score: float
    meta: dict

class HistoryRAG:
    def __init__(self, store, top_k=5, time_decay_days=30):
        self.store = store             # LanceDB / vector store wrapper you already use
        self.top_k = top_k
        self.tau = time_decay_days

    def _time_boost(self, ts: datetime) -> float:
        days = max(0, (datetime.now(timezone.utc) - ts).days)
        return math.exp(-days / self.tau)

    def retrieve(self, query: str) -> list[Hit]:
        raw = self.store.search(query, top_k=self.top_k)  # returns (text, score, meta)
        hits = []
        for text, score, meta in raw:
            ts = meta.get("timestamp") or datetime.now(timezone.utc)
            boosted = score * self._time_boost(ts)
            hits.append(Hit(text=text, ts=ts, score=boosted, meta=meta))
        return sorted(hits, key=lambda h: h.score, reverse=True)
