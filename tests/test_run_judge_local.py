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


def _run(client, max_retries=0):
    entry = {"system": "S", "user": "U"}
    sem = asyncio.Semaphore(1)
    return asyncio.run(
        rjl._judge_one(client, "judge-x", entry, 7, sem, max_retries=max_retries)
    )


class _SequenceClient:
    """AsyncOpenAI stand-in that returns a different canned response (or
    raises) on each successive call, for exercising the retry loop."""

    def __init__(self, responses: list):
        self._responses = list(responses)
        self.calls = 0

        async def create(**kwargs):
            r = self._responses[self.calls]
            self.calls += 1
            if isinstance(r, BaseException):
                raise r
            return r
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))


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


# ---------------------------------------------------------------------------
# max_retries: bounded retry then flag (2026-09-15) — narrow to EMPTY content
# + finish_reason='length'; every other no-verdict case is unaffected.
# ---------------------------------------------------------------------------


def test_length_truncation_default_still_raises():
    """max_retries defaults to 0 -- unset callers keep the original
    raise-on-first-failure behaviour exactly."""
    with pytest.raises(ValueError, match=r"no true/false verdict.*finish_reason='length'"):
        _run(_FakeClient(resp=_response(None, finish_reason="length")))


def test_length_truncation_retries_then_succeeds():
    client = _SequenceClient([
        _response(None, finish_reason="length"),
        _response(None, finish_reason="length"),
        _response("true"),
    ])
    out = _run(client, max_retries=2)
    assert out["judgement"] is True
    assert "judgement_truncated" not in out
    assert client.calls == 3


def test_length_truncation_exhausts_retries_flags_none():
    client = _SequenceClient([
        _response(None, finish_reason="length"),
        _response(None, finish_reason="length"),
        _response(None, finish_reason="length"),
    ])
    out = _run(client, max_retries=2)
    assert out["judgement"] is None
    assert out["judgement_truncated"] is True
    assert out["judgement_model"] == "judge-x"
    assert client.calls == 3  # 1 initial + 2 retries, no 4th call


def test_non_length_no_verdict_not_retried_even_with_budget():
    """A completed response (finish_reason='stop') with no true/false is a
    genuine parse failure, not a truncation -- must still raise immediately
    even when max_retries > 0, and must not consume a retry."""
    client = _SequenceClient([_response("I am not sure about this one")])
    with pytest.raises(ValueError, match=r"\[idx=7\] judge response has no true/false verdict"):
        _run(client, max_retries=2)
    assert client.calls == 1


def test_non_empty_length_truncation_not_retried():
    """finish_reason='length' with non-empty (but no true/false) text is a
    different failure than the empty-content case -- still raises
    immediately, not the retry/flag path."""
    client = _SequenceClient([_response("this got cut off mid-sent", finish_reason="length")])
    with pytest.raises(ValueError, match=r"no true/false verdict"):
        _run(client, max_retries=2)
    assert client.calls == 1


# ---------------------------------------------------------------------------
# Judge model-configs carry required max_retries/drop_ceiling (no defaults)
# ---------------------------------------------------------------------------


def test_gpt_oss_120b_has_retry_budget_and_ceiling():
    judge_cfg = rjl.paths.load_model_config("vllm_judge", "gpt-oss-120b")
    assert judge_cfg["max_retries"] == 2
    assert judge_cfg["drop_ceiling"] == 10  # absolute row count, not a fraction


@pytest.mark.parametrize("judge_key", ["llama-3.3-70b", "qwen-2.5-72b"])
def test_non_reasoning_judges_have_retry_disabled(judge_key):
    judge_cfg = rjl.paths.load_model_config("vllm_judge", judge_key)
    assert judge_cfg["max_retries"] == 0
    assert judge_cfg["drop_ceiling"] == 0


# ---------------------------------------------------------------------------
# max_concurrent / request_timeout resolution (2026-09-14: moved from
# experiment params to the judge's model-config)
# ---------------------------------------------------------------------------


def _write_config(tmp_path, config_id, extra_params: dict) -> Path:
    cfg_path = tmp_path / f"{config_id}.yaml"
    cfg_path.write_text(
        "id: " + config_id + "\n"
        "project: scholarlm\n"
        "description: test fixture\n"
        "seed: 342\n"
        "params:\n"
        "  dataset: pond\n"
        "  judge: llama-3.3-70b\n"
        "  extraction_id: does-not-matter\n"
        + "".join(f"  {k}: {v}\n" for k, v in extra_params.items())
    )
    return cfg_path


@pytest.mark.parametrize("stale_key", ["max_concurrent", "request_timeout"])
def test_main_rejects_stale_param_key(tmp_path, stale_key):
    """A config still setting max_concurrent/request_timeout in params must
    raise, not silently ignore the value and fall back to the model-config."""
    cfg_path = _write_config(tmp_path, "test-stale-param", {stale_key: 32})
    with pytest.raises(ValueError, match=r"no longer read from experiment params"):
        rjl.main([str(cfg_path)])


def test_llama_model_config_supplies_max_concurrent_and_request_timeout():
    """The values main() now resolves to must actually be present (required,
    not defaulted) on the real llama-3.3-70b model-config."""
    judge_cfg = rjl.paths.load_model_config("vllm_judge", "llama-3.3-70b")
    assert judge_cfg["max_concurrent"] == 32
    assert judge_cfg["request_timeout"] == 600
