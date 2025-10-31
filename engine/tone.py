def reframe_to_builder(text: str, style: str = "confident") -> str:
    swaps = {
        "might be hard": "is solvable by",
        "could": "will",
        "should consider": "will implement",
        "it may be possible": "we will prototype",
    }
    for a,b in swaps.items():
        text = text.replace(a, b)
    return text
