from engine.history_rag import HistoryRAG
from engine.builder_enforcer import enforce
from engine.knowledge.qa_engine import QASprint, QAGates
from engine.knowledge.store import MemoryStore
from engine.knowledge.phase_policy import PhasePolicy
import json
from typing import Dict, Any

class BuilderFirstWorkflow:
    def __init__(self, llm, vecdb, cfg):
        self.llm = llm
        self.rag = HistoryRAG(vecdb, top_k=cfg["history"]["top_k"],
                              time_decay_days=cfg["history"]["time_decay_days"])
        self.cfg = cfg

    def _builder_prompt(self, user_prompt: str, history_hits: list, qa_context: dict = None):
        hist = "\n\n".join([f"- ({h.ts.date()}) {h.text[:280]}" for h in history_hits])
        qa_info = f"QA METRICS: {qa_context}" if qa_context else ""
        return f"""
You are a senior systems builder. For EVERY challenge you list, propose ≥ {self.cfg['solutions_per_problem']} solution pathways.
Always end with a Week 1 deliverable (files, commands, tests).

User prompt:
{user_prompt}

Relevant prior context:
{hist}

{qa_info}

Use the following structure strictly:
{open('prompts/builder_pack.md').read()}
"""

    def _run_qa_sprint(self, synopsis: str, phase: str) -> Dict[str, Any]:
        store = MemoryStore()
        gates = QAGates()
        policy = PhasePolicy(phase=phase)

        def llm_call(prompt: str) -> str:
            return self.llm(prompt)

        def judge_call(node) -> dict:
            critique = self.llm(f"""
Score this {node.kind} for quality (0..1) and confidence (0..1).
TEXT: {node.content}
Return JSON: {{"quality": <0..1>,"confidence": <0..1>,"evidence":[]}}
""")
            try:
                return json.loads(critique)
            except:
                return {"quality":0.6,"confidence":node.confidence,"evidence":[]}

        sprint = QASprint(llm_call, judge_call, store, gates)
        m = sprint.run(synopsis, max_rounds=1)
        return m | policy.check(m)

    def run(self, user_prompt: str, phase: str = "inception"):
        # Run a QA sprint first
        qa_metrics = self._run_qa_sprint(user_prompt, phase)

        hits = self.rag.retrieve(user_prompt)
        prompt = self._builder_prompt(user_prompt, hits, qa_context=qa_metrics)

        draft = self.llm(prompt)  # your local model / API wrapper
        def regen_fn(p): return self.llm(p)

        final, report = enforce(prompt, draft, self.cfg, regen_fn)

        # Add QA metrics to the final report
        report["qa_metrics"] = qa_metrics

        return final, report
