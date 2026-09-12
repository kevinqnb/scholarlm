"""Rung-1 unit tests for ``run_judge_local._judge_one`` fail-loud behaviour.

A judge request that errors, returns no content, or produces no true/false
verdict must raise — never resolve to a null judgement that ``run_judge_combine``
would silently fold into the majority vote.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "experiments"))

import run_judge_local as rjl


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def _response(content, *, finish_reason="stop", model="judge-x", prompt_tokens=42):
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=content),
            finish_reason=finish_reason,
        )],
        model=model,
        usage=SimpleNamespace(prompt_tokens=prompt_tokens),
    )


class _FakeClient:
    """Minimal AsyncOpenAI stand-in: ``chat.completions.create`` returns
    ``resp`` or raises ``exc``."""

    def __init__(self, *, resp=None, exc=None):
        async def create(**kwargs):
            if exc is not None:
                raise exc
            return resp
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))


def _run(client):
    entry = {"system": "S", "user": "U"}
    sem = asyncio.Semaphore(1)
    return asyncio.run(rjl._judge_one(client, "judge-x", entry, 7, sem))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text,expected", [
    ("true", True),
    ("false", False),
    ("  True.\n", True),
    ("false\n\n", False),
])
def test_clean_verdict_parses(text, expected):
    out = _run(_FakeClient(resp=_response(text)))
    assert out["judgement"] is expected
    assert out["judgement_model"] == "judge-x"
    assert out["_prompt_tokens"] == 42


def test_no_verdict_raises():
    with pytest.raises(ValueError, match=r"\[idx=7\] judge response has no true/false verdict"):
        _run(_FakeClient(resp=_response("I am not sure about this one")))


def test_empty_content_raises():
    with pytest.raises(ValueError, match=r"no true/false verdict.*finish_reason='length'"):
        _run(_FakeClient(resp=_response(None, finish_reason="length")))


def test_api_error_raises_with_index():
    with pytest.raises(RuntimeError, match=r"\[idx=7\] judge API call failed: TimeoutError"):
        _run(_FakeClient(exc=TimeoutError("read timed out")))


def test_no_verdict_does_not_resolve_to_none():
    """The failure must be an exception, not a dict with judgement=None."""
    try:
        _run(_FakeClient(resp=_response("maybe")))
    except ValueError:
        return
    pytest.fail("expected _judge_one to raise, not return")
