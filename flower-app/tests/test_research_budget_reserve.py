"""Concurrent gathering cannot spend the calls reserved for a final review."""

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from fusion_agent.research.engine import ModelCallBudgetReserved, ResearchEngine


def test_parallel_collectors_cannot_race_past_the_review_call_reserve():
    class Model:
        def complete(self, messages, tools, tool_choice):
            return {"role": "assistant", "content": "Offline fixture."}, {"total_tokens": 1}

    engine = ResearchEngine([], Model(), None, lambda _event: None, max_turns=10)
    barrier = threading.Barrier(8)

    def collect(_index):
        barrier.wait(timeout=5)
        try:
            engine.request([], [], reserve_calls=7)
            return "called"
        except ModelCallBudgetReserved:
            return "reserved"

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(collect, range(8)))
    assert results.count("called") == 3
    assert results.count("reserved") == 5
    assert engine.model_calls == 3
    for _ in range(7):
        engine.request([], [])
    assert engine.model_calls == 10
    with pytest.raises(RuntimeError, match="model-call budget reached"):
        engine.request([], [])
