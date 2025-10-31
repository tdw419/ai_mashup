import re
from dataclasses import dataclass

@dataclass
class GateReport:
    problems: int
    solutions: int
    ratio: float
    actionability: float
    specificity: float
    vague_ratio: float
    passed: bool
    notes: list[str]

VAGUE_PATTERNS = [
    r"\b(it|this|that|they|these)\s+(might|could|may)\b",
    r"\bshould consider\b", r"\bto be determined\b", r"\bTBD\b"
]

def count_bullets(text: str, prefix="- "):
    return sum(1 for line in text.splitlines() if line.strip().startswith(prefix))

def measure_actionability(text: str) -> float:
    # heuristics: presence of imperative verbs + concrete artifacts
    hits = len(re.findall(r"\b(implement|create|run|ship|test|deploy|write|hook|add)\b", text, re.I))
    artifacts = len(re.findall(r"\b(PR|branch|file|script|endpoint|test|command)\b", text, re.I))
    return min(1.0, (hits + 0.5*artifacts) / 6)

def measure_specificity(text: str) -> float:
    numbers = len(re.findall(r"\b\d+(\.\d+)?\b", text))
    code_blocks = text.count("```")
    paths = len(re.findall(r"[a-zA-Z0-9_\-./]+/[a-zA-Z0-9_\-./]+", text))
    return min(1.0, (0.6*numbers + 0.8*code_blocks + 0.4*paths) / 5)

def vague_ratio(text: str) -> float:
    tokens = max(1, len(text.split()))
    vague = 0
    for pat in VAGUE_PATTERNS:
        vague += len(re.findall(pat, text, re.I))
    return min(1.0, vague / max(50, tokens/8))

def audit(text: str, cfg) -> GateReport:
    problems = len(re.findall(r"(?i)challenge|problem|risk|limitation", text))
    solutions = len(re.findall(r"(?i)solution|approach|mitigation|pathway|fix", text))
    ratio = (solutions / max(1, problems)) if problems else 2.0
    act = measure_actionability(text)
    spec = measure_specificity(text)
    vr = vague_ratio(text)
    ok = (ratio >= cfg["solutions_per_problem"]
          and act >= cfg["quality_gates"]["min_actionability"]
          and spec >= cfg["quality_gates"]["min_specificity"]
          and vr <= cfg["quality_gates"]["max_vague_ratio"])
    notes = []
    if ratio < cfg["solutions_per_problem"]: notes.append("Increase solutions/problem ratio.")
    if act  < cfg["quality_gates"]["min_actionability"]: notes.append("Make steps executable.")
    if spec < cfg["quality_gates"]["min_specificity"]: notes.append("Add numbers/files/branches.")
    if vr   > cfg["quality_gates"]["max_vague_ratio"]: notes.append("Reduce hedging language.")
    return GateReport(problems, solutions, ratio, act, spec, vr, ok, notes)

def enforce(prompt: str, draft: str, cfg, regen_fn):
    report = audit(draft, cfg)
    if report.passed:
        return draft, report
    correction = f"""
The draft below fails builder requirements.

PROMPT:
{prompt}

DRAFT:
{draft}

FIX IT. Requirements:
- Pair EVERY challenge with ≥ {cfg['solutions_per_problem']} solution pathways.
- Add a **Week 1 deliverable** with files/commands.
- Replace hedging with decisive builder framing.
- Include concrete artifacts (files, scripts, branches, tests).

Return the corrected response.
"""
    best = draft
    for _ in range(cfg.get("deep_think_on_fail", 1)):
        best = regen_fn(correction)
        r2 = audit(best, cfg)
        if r2.passed: return best, r2
    return best, audit(best, cfg)
