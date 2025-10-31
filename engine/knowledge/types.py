from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime
from typing import List, Dict, Any, Optional

class KState(Enum):
    RAW_QUESTION = "raw_question"
    ACTIVE_INVESTIGATION = "active_investigation"
    PARTIALLY_ANSWERED = "partially_answered"
    CONFIDENT_ANSWER = "confident_answer"
    VALIDATED = "validated"

@dataclass
class KNode:
    id: str
    kind: str            # "question" | "answer"
    content: str
    state: KState
    confidence: float    # 0..1
    deps: List[str]      # upstream node ids
    evidence: List[str]  # paths, tests, refs
    created_at: datetime
    updated_at: datetime
    quality: float = 0.0 # question_quality or answer_quality

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["state"] = self.state.value
        return d
