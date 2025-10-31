from engine.builder_enforcer import audit

def test_builder_gate_passes_on_good_draft():
    cfg = {
        "solutions_per_problem": 1.5,
        "quality_gates": {
            "min_actionability": 0.75,
            "min_specificity": 0.70,
            "max_vague_ratio": 0.30
        }
    }
    draft = """
## Challenges & Solutions
- Challenge: command routing in the pixel VM
  → Solution A: table-driven opcode map (src/vm/op.rs), gen via build.rs
  → Solution B: derived macro with enum-dispatch (benchmark both)
  → Solution C: auto-generated jump table from DSL (future)

## Implementation Roadmap
- Week 1:
  Files: src/vm/op.rs, src/bin/pxsh.rs, tests/opcode_table.rs
  Command(s): cargo add enum_dispatch && cargo test -q
  Test: ensure 16 ops resolve in O(1); include bench

## Open Questions
- Is macro dispatch measurably faster on 5090?
"""
    rep = audit(draft, cfg)
    assert rep.passed, rep
