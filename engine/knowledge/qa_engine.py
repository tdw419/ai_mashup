from typing import List, Dict, Any
from datetime import datetime
from .types import KNode, KState, KState as S
from .store import MemoryStore, new_question, new_answer

class QAGates:
    def __init__(self, answer_threshold=0.8, min_q_quality=0.5, min_a_quality=0.7):
        self.answer_threshold = answer_threshold
        self.min_q_quality = min_q_quality
        self.min_a_quality = min_a_quality

    def question_ok(self, q: KNode) -> bool:
        return q.quality >= self.min_q_quality

    def answer_ok(self, a: KNode) -> bool:
        return a.confidence >= self.answer_threshold and a.quality >= self.min_a_quality

class QASprint:
    """Turns uncertainty into questions and certainty into answers."""

    def __init__(self, llm_call, judge_call, store: MemoryStore, gates: QAGates):
        self.llm = llm_call         # (prompt:str) -> str/json
        self.judge = judge_call     # (node:KNode) -> dict with {confidence,quality,evidence}
        self.store = store
        self.gates = gates

    # --- public API ---
    def run(self, synopsis: str, max_rounds=2) -> Dict[str, Any]:
        created, updated = [], []
        for _ in range(max_rounds):
            qs = self._gen_questions(synopsis)
            created += qs
            ans = self._attempt_answers(qs)
            created += ans
            self._promote_states(qs + ans, updated)
        if created:
            self.store.add(created + updated)
        return self._metrics(created + updated)

    # --- internals ---
    def _gen_questions(self, synopsis: str) -> List[KNode]:
        prompt = f"""
Generate 5-8 *high-leverage* engineering questions that, if answered, unlock progress.
Return as bullet points, terse, implementation-leaning.
CONTEXT: {synopsis}
"""
        text = self.llm(prompt)
        lines = [l.strip("-* ").strip() for l in text.splitlines() if l.strip()]
        nodes = []
        for line in lines[:8]:
            q = new_question(line, deps=[], quality=self._score_question(line))
            if self.gates.question_ok(q):
                nodes.append(q)
        return nodes

    def _attempt_answers(self, questions: List[KNode]) -> List[KNode]:
        answers = []
        for q in questions:
            atext = self.llm(f"""
Question: {q.content}
Give the *best concrete answer you can now*.
Format:
ANSWER: ...
EVIDENCE: file paths, tests, commands or rationale
CONFIDENCE: 0..1
""")
            # quick parse
            conf = self._extract_float(atext, "CONFIDENCE", default=0.5)
            ev = self._extract_section(atext, "EVIDENCE")
            ans = new_answer(
                content=self._extract_section(atext, "ANSWER") or atext,
                deps=[q.id], confidence=conf, evidence=ev.splitlines() if ev else []
            )
            scored = self.judge(ans)  # adds confidence/quality/evidence if judges
            ans.confidence = max(ans.confidence, scored.get("confidence", ans.confidence))
            ans.quality = scored.get("quality", 0.0)
            if self.gates.answer_ok(ans):
                ans.state = S.CONFIDENT_ANSWER
            answers.append(ans)
        return answers

    def _promote_states(self, nodes: List[KNode], updated_out: List[KNode]):
        for n in nodes:
            if n.kind == "answer" and n.state == S.CONFIDENT_ANSWER and self._has_test_evidence(n):
                n.state = S.VALIDATED
            n.updated_at = datetime.utcnow()
            updated_out.append(n)

    # --- helpers ---
    def _score_question(self, line: str) -> float:
        # crude: longer + technical tokens score higher
        hits = sum(t in line.lower() for t in [
            "interface","latency","throughput","cache","opcode",
            "benchmark","test","api","schema","memory","gpu","wgsl","rust"
        ])
        return min(1.0, 0.3 + 0.1*hits + 0.02*len(line))

    def _has_test_evidence(self, a: KNode) -> bool:
        joined = " ".join(a.evidence).lower()
        for tok in ["tests/", "cargo test", "pytest", "integration", "bench", "ci"]:
            if tok in joined: return True
        return False

    def _extract_section(self, text: str, key: str) -> str:
        # naive but robust: try "KEY:" then until next ALLCAPS or end
        import re
        m = re.search(rf"{key}:\s*(.+)", text, re.IGNORECASE|re.DOTALL)
        return m.group(1).strip() if m else ""

    def _extract_float(self, text: str, key: str, default=0.5) -> float:
        import re
        m = re.search(rf"{key}:\s*([01](?:\.\d+)?)", text, re.IGNORECASE)
        try: return float(m.group(1)) if m else default
        except: return default

    def _metrics(self, nodes: List[KNode]) -> Dict[str, Any]:
        q = [n for n in nodes if n.kind=="question"]
        a = [n for n in nodes if n.kind=="answer"]
        conf = [n for n in a if n.state==S.CONFIDENT_ANSWER]
        val = [n for n in a if n.state==S.VALIDATED]
        return {
            "created_questions": len(q),
            "created_answers": len(a),
            "confident_answers": len(conf),
            "validated_answers": len(val),
            "qa_ratio": (len(a) / max(1,len(q))),
        }
