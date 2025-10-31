from engine.knowledge.phase_policy import PhasePolicy

def test_inception_prefers_questions():
    p = PhasePolicy("inception")
    # simulate 2.5 questions per answer → answers/questions = 0.4
    m = {"qa_ratio": 0.4}
    chk = p.check(m)
    assert chk["ok_questions_per_answer"] and not chk["ok_answers_per_question"]

def test_delivery_prefers_answers():
    p = PhasePolicy("delivery")
    # simulate 3 answers per question → qa_ratio = 3.0
    m = {"qa_ratio": 3.0}
    chk = p.check(m)
    assert chk["ok_answers_per_question"] and not chk["ok_questions_per_answer"]
