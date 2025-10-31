#!/usr/bin/env python3
import argparse, json
from engine.knowledge.qa_engine import QASprint, QAGates
from engine.knowledge.store import MemoryStore
from engine.knowledge.phase_policy import PhasePolicy

def dummy_llm(prompt:str)->str:
    # Swap with your LLM wrapper
    return "ANSWER: Draft implementation path\nEVIDENCE: tests/test_x.py\nCONFIDENCE: 0.62"

def dummy_judge(node)->dict:
    return {"quality":0.72, "confidence":max(0.62, node.confidence), "evidence":node.evidence}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synopsis", required=True)
    ap.add_argument("--phase", choices=["inception","delivery"], default="inception")
    args = ap.parse_args()

    gates = QAGates()
    store = MemoryStore()
    sprint = QASprint(dummy_llm, dummy_judge, store, gates)
    m = sprint.run(args.synopsis, max_rounds=1)
    policy = PhasePolicy(args.phase).check(m)
    print(json.dumps(m | policy, indent=2))

if __name__ == "__main__":
    main()
