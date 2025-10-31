class PhasePolicy:
    """
    Switch behavior by project phase:
      - "inception": encourage ≥ 2.0 questions per answer
      - "delivery": encourage ≥ 2.0 answers per question
    """
    def __init__(self, phase: str = "inception"):
        self.phase = phase

    def target_ratios(self):
        if self.phase == "inception":
            return {"min_Q_per_A": 2.0, "min_A_per_Q": 0.5}
        if self.phase == "delivery":
            return {"min_Q_per_A": 0.5, "min_A_per_Q": 2.0}
        return {"min_Q_per_A": 1.0, "min_A_per_Q": 1.0}

    def check(self, metrics: dict) -> dict:
        qa = metrics.get("qa_ratio", 1.0)           # answers / questions
        inv = 1.0 / qa if qa > 0 else 99.0          # questions / answers
        t = self.target_ratios()
        return {
            "ok_questions_per_answer": inv >= t["min_Q_per_A"],
            "ok_answers_per_question": qa >= t["min_A_per_Q"],
            "phase": self.phase,
            "observed_Q_per_A": inv,
            "observed_A_per_Q": qa
        }
