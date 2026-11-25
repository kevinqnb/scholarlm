"""Unit tests for run_judge_baselines.py.

A fake vLLM server (httpx.MockTransport) sits behind the real AsyncOpenAI client, so
requests and responses go through the actual wire format.
"""
from __future__ import annotations

import asyncio
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from openai import AsyncOpenAI, InternalServerError

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "experiments"))

import run_judge_baselines as rjb
import run_judge_combine as rjc
from scholarlm.instruction_prompts import JUDGE_CONFIDENCE_INSTRUCTIONS, JUDGE_INSTRUCTIONS
from scholarlm.utils import judge_prompts

LABEL_IDS = {"true": 11, "True": 12, "false": 13, "False": 14}
SP = {"temperature": 0.0, "top_p": 1.0, "top_k": -1, "min_p": 0.0, "repetition_penalty": 1.0}
VERB = {"base_seed": 342, "sampling_params": SP, "retry_temperature": 0.7, "max_tokens": 16,
        "max_retries": 2, "reasoning_effort": None}
SAMP = {"base_seed": 342, "sampling_params": SP | {"temperature": 1.0}, "max_tokens": 16,
        "max_retries": 2, "reasoning_effort": None}
ENTRY = {"system": "S", "user": "U", "context_text": "C", "user_query": "Q"}


class CharTokenizer:
    """One token per character, except the four labels."""
    def apply_chat_template(self, chat, tokenize, add_generation_prompt):
        return "<u>" + chat[0]["content"] + "<a>"

    def __call__(self, text, return_offsets_mapping, add_special_tokens):
        return {"input_ids": [ord(c) for c in text],
                "offset_mapping": [(i, i + 1) for i in range(len(text))]}

    def encode(self, s, add_special_tokens):
        return [LABEL_IDS[s]]

    def decode(self, ids):
        return {v: k for k, v in LABEL_IDS.items()}.get(ids[0], "other")


def chat_body(text, finish_reason="stop"):
    return {
        "id": "x", "object": "chat.completion", "created": 0, "model": "m",
        "choices": [{"index": 0, "finish_reason": finish_reason,
                     "message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": 7, "completion_tokens": 1, "total_tokens": 8},
    }


def run(handler, scorer, *args, **kwargs):
    async def go():
        async with AsyncOpenAI(
            api_key="EMPTY", base_url="http://vllm.test/v1", max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        ) as client:
            return await scorer(client, "m", ENTRY, 3, asyncio.Semaphore(4), *args, **kwargs)
    return asyncio.run(go())


def chat_server(texts):
    """Serve ``texts`` in order ((text, finish_reason) or text); record request bodies."""
    seen = []

    def handler(request):
        assert request.url.path == "/v1/chat/completions"
        item = texts[len(seen)]
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=chat_body(*([item] if isinstance(item, (str, type(None))) else item)))
    return handler, seen


# ---------------------------------------------------------------------------
# Parsing / prompts / seeds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text,expected", [
    ("0", 0.0), ("100", 1.0), ("85", 0.85), ("85%", 0.85), ("85.", 0.85), ("1", 0.01),
    ("80\n\nThe extracted value of 1.2 m", 0.8), ("34 \n\nThe value", 0.34), ("85%\nBecause", 0.85),
    ("80.5\nBecause", None), ("80 because", None), ("80\tbecause", None), ("150\n\nThe", None),
    ("1000\n", None), ("Confidence\n80", None),
    ("0.85", None), ("0.5", None), ("72.5", None), ("1.0", None),
    ("101", None), ("-5", None), ("", None), ("85 or 90", None), ("Confidence: 85", None),
    ("true", None), ("0.85.1", None),
])
def test_parse_confidence(text, expected):
    assert rjb.parse_confidence(text) == expected


@pytest.mark.parametrize("text,expected", [
    ("true", True), ("True.", True), ("FALSE", False), ("", None), ("maybe", None),
    ("false, it is not true", False), ("**True**", True), ("true. The value is in Table 2", True),
    ("not true", None), ("untrue", None), ("trueish", None), ("The answer is true", None),
])
def test_parse_verdict(text, expected):
    assert rjb.parse_verdict(text) is expected


def test_confidence_prompt_keeps_the_criteria_verbatim():
    def criteria(s):
        return s[s.index("(A) "):s.index("(F) ")] + s[s.index("(F) "):].split("\n")[0]
    assert criteria(JUDGE_CONFIDENCE_INSTRUCTIONS) == criteria(JUDGE_INSTRUCTIONS)
    for leftover in ("'true'", "'false'", "true or false"):
        assert leftover not in JUDGE_CONFIDENCE_INSTRUCTIONS


def _real_entry():
    query = judge_prompts.build_judge_query(
        attribute_description="a", entity_type_description="t",
        entity_description={"name": "n"}, measurement_val=1,
    )
    return {"custom_id": "0", "system": JUDGE_INSTRUCTIONS, "user_query": query,
            "context_text": "doc"}


def test_confidence_entries_change_only_instructions_and_closing():
    e = _real_entry()
    (new,) = rjb.to_confidence_entries([e], SimpleNamespace(name="d", judge_instructions=None))
    assert new["system"] == JUDGE_CONFIDENCE_INSTRUCTIONS
    assert new["user_query"] == e["user_query"].replace(rjb.VERDICT_CLOSING, rjb.CONFIDENCE_CLOSING)
    assert new["context_text"] == e["context_text"] and "true or false" not in rjb.user_message(new)


def test_confidence_entries_reject_custom_instructions_and_unknown_layout():
    with pytest.raises(ValueError, match="custom judge_instructions"):
        rjb.to_confidence_entries([_real_entry()], SimpleNamespace(name="d", judge_instructions="X"))
    with pytest.raises(ValueError, match="layout"):
        rjb.to_confidence_entries([_real_entry() | {"user_query": "Is it?"}],
                                  SimpleNamespace(name="d", judge_instructions=None))


def test_user_message_is_the_message_the_ntp_prompt_tokenizes():
    ids = rjb.tokenize(ENTRY["system"], ENTRY["context_text"], ENTRY["user_query"], CharTokenizer())[0]
    assert "".join(map(chr, ids)) == f"<u>{rjb.user_message(ENTRY)}<a>"


def test_sample_seeds_distinct_and_deterministic():
    seeds = [rjb.sample_seed(342, i, k, a) for i in range(20) for k in range(10) for a in range(2)]
    assert len(set(seeds)) == len(seeds)
    assert seeds[0] == rjb.sample_seed(342, 0, 0, 0)
    assert all(0 <= s < 2**31 for s in seeds)


# ---------------------------------------------------------------------------
# verbalized
# ---------------------------------------------------------------------------


def test_verbalized_scores_and_sends_every_sampling_param():
    handler, seen = chat_server(["85"])
    out = run(handler, rjb.score_verbalized, **VERB)
    assert out["score"] == 0.85 and out["raw_text"] == "85" and out["rejected"] == []
    (body,) = seen
    assert {k: body[k] for k in rjb.SAMPLING_KEYS} == SP
    assert body["max_tokens"] == 16 and body["seed"] == rjb.sample_seed(342, 3, 0, 0)
    assert body["messages"] == [{"role": "user", "content": rjb.user_message(ENTRY)}]
    assert "chat_template_kwargs" not in body


@pytest.mark.parametrize("bad", ["150", "-5", "0.85", "fairly confident", (None, "length")])
def test_verbalized_unusable_answer_is_retried_at_retry_temperature(bad):
    handler, seen = chat_server([bad, "40"])
    out = run(handler, rjb.score_verbalized, **VERB)
    assert out["score"] == 0.4 and len(out["rejected"]) == 1
    assert [b["temperature"] for b in seen] == [0.0, 0.7]
    assert seen[0]["seed"] != seen[1]["seed"]


def test_verbalized_raises_after_retry_budget():
    handler, seen = chat_server(["150", "200", "999"])
    with pytest.raises(rjb.RowFailure, match=r"\[idx=3, sample=0\] no usable answer in 3 attempts") as ei:
        run(handler, rjb.score_verbalized, **VERB)
    assert ei.value.detail == {"rejected": ["150", "200", "999"]}
    assert len(seen) == 3


def test_zero_retries_raises_on_first_unusable_answer():
    handler, seen = chat_server(["150"])
    with pytest.raises(rjb.RowFailure, match="1 attempts"):
        run(handler, rjb.score_verbalized, **VERB | {"max_retries": 0})
    assert len(seen) == 1


def test_reasoning_effort_reaches_the_chat_template():
    handler, seen = chat_server(["85"])
    run(handler, rjb.score_verbalized, **VERB | {"reasoning_effort": "medium"})
    assert seen[0]["chat_template_kwargs"] == {"reasoning_effort": "medium"}


@pytest.mark.parametrize("scorer,kwargs", [
    (rjb.score_verbalized, VERB), (rjb.score_sampling, SAMP | {"n_samples": 3}),
    (rjb.score_ntp, {"tokenizer": CharTokenizer(), "label_ids": LABEL_IDS, "top_logprobs": 20})])
def test_request_error_is_not_a_row_failure(scorer, kwargs):
    with pytest.raises(InternalServerError):
        run(lambda r: httpx.Response(500, json={}), scorer, **kwargs)


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------


def test_sampling_score_is_fraction_true():
    handler, seen = chat_server(["true"] * 7 + ["false"] * 2 + ["False."])
    out = run(handler, rjb.score_sampling, n_samples=10, **SAMP)
    assert out["score"] == 0.7 and len(out["raw_text"]) == 10 and out["n_nonliteral"] == 1
    assert len({b["seed"] for b in seen}) == 10
    assert all({k: b[k] for k in rjb.SAMPLING_KEYS} == SP | {"temperature": 1.0} for b in seen)


def _stuck_server(stuck_seeds):
    """Answers "maybe" to the given seeds and "true" to all others."""
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        return httpx.Response(200, json=chat_body("maybe" if body["seed"] in stuck_seeds else "true"))
    return handler, seen


def test_sampling_retries_only_the_unusable_sample():
    handler, seen = _stuck_server({rjb.sample_seed(342, 3, 1, 0), rjb.sample_seed(342, 3, 1, 1)})
    out = run(handler, rjb.score_sampling, n_samples=3, **SAMP)
    assert out["score"] == 1.0 and out["rejected"] == ["maybe", "maybe"]
    assert len(out["raw_text"]) == 3 and len(seen) == 5
    assert all(b["temperature"] == 1.0 for b in seen)


def test_sampling_raises_when_one_sample_exhausts_its_budget():
    handler, _ = _stuck_server({rjb.sample_seed(342, 3, 1, a) for a in range(3)})
    with pytest.raises(rjb.RowFailure, match=r"1/3 samples failed.*sample=1\] no usable answer") as ei:
        run(handler, rjb.score_sampling, n_samples=3, **SAMP)
    samples = ei.value.detail["samples"]
    assert samples[0] == samples[2] == "true" and samples[1]["detail"] == {"rejected": ["maybe"] * 3}


# ---------------------------------------------------------------------------
# ntp
# ---------------------------------------------------------------------------


def test_ntp_known_answer():
    lp = {"true": math.log(0.5), "True": math.log(0.1), "false": math.log(0.15), "False": math.log(0.05)}
    p_true, p_false, mass = rjb.ntp_from_logprobs(lp)
    assert p_true == pytest.approx(0.75) and p_false == pytest.approx(0.25) and mass == pytest.approx(0.8)


def test_ntp_matches_softmax_over_logsumexp_of_logits():
    torch = pytest.importorskip("torch")
    logits = torch.randn(500, dtype=torch.float64, generator=torch.Generator().manual_seed(0)) * 4
    ids = [3, 77, 200, 444]
    pair = torch.stack([torch.logsumexp(logits[ids[:2]], 0), torch.logsumexp(logits[ids[2:]], 0)])
    expected = torch.softmax(pair, 0)
    lps = torch.log_softmax(logits, 0)[ids].tolist()
    p_true, p_false, _ = rjb.ntp_from_logprobs(dict(zip(rjb.LABELS, lps)))
    assert p_true == pytest.approx(expected[0].item(), rel=1e-12)
    assert p_false == pytest.approx(expected[1].item(), rel=1e-12)


def ntp_server(probs, top, *, prompt_token_delta=0):
    """Fake vLLM: ``probs`` is the next-token distribution over ids, ``top`` the ids in the top 20."""
    seen = {"completions": [], "scoring": []}

    def handler(request):
        body = json.loads(request.content)
        if request.url.path == "/v1/completions":
            seen["completions"].append(body)
            n = len(body["prompt"]) + prompt_token_delta
            return httpx.Response(200, json={
                "id": "x", "object": "text_completion", "created": 0, "model": "m",
                "choices": [{"index": 0, "text": "true", "finish_reason": "length", "logprobs": {
                    "tokens": ["token_id:11"], "token_logprobs": [math.log(probs[11])], "text_offset": [0],
                    "top_logprobs": [{f"token_id:{t}": math.log(probs[t]) for t in top}]}}],
                "usage": {"prompt_tokens": n, "completion_tokens": 1, "total_tokens": n + 1}})
        assert request.url.path == "/generative_scoring"
        seen["scoring"].append(body)
        n = len(body["query"]) + len(body["items"][0]) + prompt_token_delta
        return httpx.Response(200, json={
            "id": "s", "object": "list", "created": 0, "model": "m",
            "data": [{"index": 0, "object": "score", "score": probs[body["label_token_ids"][0]]}],
            "usage": {"prompt_tokens": n, "total_tokens": n + 1}})
    return handler, seen


PROBS = {11: 0.5, 12: 0.1, 13: 0.15, 14: 0.05, 99: 0.2}


def test_ntp_single_forward_when_all_labels_in_top():
    handler, seen = ntp_server(PROBS, [11, 12, 13, 14, 99])
    out = run(handler, rjb.score_ntp, tokenizer=CharTokenizer(), label_ids=LABEL_IDS, top_logprobs=20)
    assert out["score"] == pytest.approx(0.75) and out["label_mass"] == pytest.approx(0.8)
    assert out["labels_looked_up"] == [] and seen["scoring"] == []
    (body,) = seen["completions"]
    # One user message, templated, no extra special tokens.
    assert body["prompt"] == [ord(c) for c in "<u>## INSTRUCTIONS:\nS\n\n## CONTEXT:\nC\n\n## QUERY:\nQ<a>"]
    assert body["max_tokens"] == 1 and body["logprobs"] == 20 and body["return_tokens_as_token_ids"] is True
    assert out["raw_text"] == "true" and out["top_token_is_label"]


def test_ntp_top_token_outside_labels_is_recorded():
    probs = {11: 0.01, 12: 0.01, 13: 0.01, 14: 0.01, 99: 0.96}
    handler, _ = ntp_server(probs, [11, 12, 13, 14, 99])
    out = run(handler, rjb.score_ntp, tokenizer=CharTokenizer(), label_ids=LABEL_IDS, top_logprobs=20)
    assert out["raw_text"] == "other" and not out["top_token_is_label"]
    assert out["label_mass"] == pytest.approx(0.04)


def test_ntp_label_outside_top_is_looked_up_exactly():
    handler, seen = ntp_server(PROBS, [11, 13, 99])
    out = run(handler, rjb.score_ntp, tokenizer=CharTokenizer(), label_ids=LABEL_IDS, top_logprobs=20)
    assert out["score"] == pytest.approx(0.75)
    assert out["labels_looked_up"] == ["True", "False"]
    assert [b["label_token_ids"] for b in seen["scoring"]] == [[12], [14]]
    prompt = seen["completions"][0]["prompt"]
    for b in seen["scoring"]:
        assert b["query"] + b["items"][0] == prompt and b["apply_softmax"] is False


def test_ntp_tokenizes_only_after_acquiring_the_semaphore():
    async def go():
        sem = asyncio.Semaphore(1)
        await sem.acquire()
        # tokenizer=None would raise as soon as tokenization is attempted.
        task = asyncio.create_task(rjb.score_ntp(
            None, "m", ENTRY, 3, sem, tokenizer=None, label_ids=LABEL_IDS, top_logprobs=20))
        await asyncio.sleep(0.01)
        assert not task.done()
        task.cancel()
    asyncio.run(go())


def test_ntp_prompt_length_mismatch_raises():
    handler, _ = ntp_server(PROBS, [11, 12, 13, 14], prompt_token_delta=1)
    with pytest.raises(ValueError, match=r"\[idx=3\].*prompt tokens"):
        run(handler, rjb.score_ntp, tokenizer=CharTokenizer(), label_ids=LABEL_IDS, top_logprobs=20)


def test_label_token_ids_rejects_multi_token_labels():
    with pytest.raises(ValueError, match="not a single token"):
        rjb.label_token_ids(SimpleNamespace(encode=lambda s, add_special_tokens: [1, 2]))
    with pytest.raises(ValueError, match="not distinct"):
        rjb.label_token_ids(SimpleNamespace(encode=lambda s, add_special_tokens: [1]))


# ---------------------------------------------------------------------------
# Params
# ---------------------------------------------------------------------------

_COMMON = {"dataset": "pond", "judge": "llama-3.1-8b", "ocr_dir": "o", "extraction_id": "x"}
VALID = {
    "verbalized": _COMMON | {"baseline": "verbalized", "max_tokens": 16, "max_retries": 2,
                             "sampling_params": SP, "retry_temperature": 0.7},
    "sampling": _COMMON | {"baseline": "sampling", "max_tokens": 16, "max_retries": 2,
                           "sampling_params": SP | {"temperature": 1.0}, "n_samples": 10},
    "ntp": _COMMON | {"baseline": "ntp", "top_logprobs": 20, "walltime": "1:00:00"},
}


@pytest.mark.parametrize("baseline", VALID)
def test_valid_params(baseline):
    rjb.validate_params(VALID[baseline], {}, "cfg")


@pytest.mark.parametrize("baseline,key", [
    (b, k) for b, p in VALID.items() for k in p if k != "walltime"])
def test_every_param_is_required(baseline, key):
    with pytest.raises(ValueError):
        rjb.validate_params({k: v for k, v in VALID[baseline].items() if k != key}, {}, "cfg")


@pytest.mark.parametrize("baseline,override,match", [
    ("ntp", {"baseline": "nope"}, "must be one of"),
    ("ntp", {"synthetic_file": "f"}, "exactly one"),
    ("ntp", {"max_retries": 2}, "not used by baseline"),
    ("verbalized", {"max_concurrent": 4}, "not used by baseline"),
    ("verbalized", {"retry_temperature": 0}, "greedy retry"),
    ("verbalized", {"sampling_params": {"temperature": 0.0}}, "exactly the keys"),
    ("verbalized", {"max_retries": -1}, "max_retries must be"),
    ("sampling", {"sampling_params": SP}, "temperature > 0"),
    ("sampling", {"n_samples": 1}, "n_samples must be"),
    ("ntp", {"row_indices": []}, "row_indices"), ("ntp", {"row_indices": [1, 1]}, "row_indices"),
    ("ntp", {"row_indices": [-1]}, "row_indices"), ("ntp", {"row_indices": 5}, "row_indices"),
])
def test_invalid_params(baseline, override, match):
    with pytest.raises(ValueError, match=match):
        rjb.validate_params(VALID[baseline] | override, {}, "cfg")


def test_ntp_rejects_reasoning_judge():
    cfg = rjb.paths.load_model_config("vllm_judge", "gpt-oss-120b")
    with pytest.raises(ValueError, match="reasoning judge"):
        rjb.validate_params(VALID["ntp"], cfg, "cfg")


def test_experiment_type_registered():
    info = rjb.paths.EXPERIMENT_TYPES["judge_baselines"]
    assert info == {"runner": "run_judge_baselines.py", "model_kind": "vllm_judge", "model_param": "judge"}


# ---------------------------------------------------------------------------
# End to end through run_judge_baseline
# ---------------------------------------------------------------------------


@pytest.fixture
def e2e(tmp_path, monkeypatch):
    """Two rows out of document order; the fake judge answers by document."""
    rows = [{"measurement_id": 10, "document_id": "b"}, {"measurement_id": 11, "document_id": "a"}]
    inp = tmp_path / "final.json"
    inp.write_text(json.dumps(rows))

    def entry(idx, doc):
        return {"custom_id": str(idx), "document_id": doc, "system": JUDGE_INSTRUCTIONS,
                "user_query": f"q {rjb.VERDICT_CLOSING}", "context_text": doc}

    answers = {"a": ("10", "stop"), "b": ("90", "stop")}

    def handler(request):
        (message,) = json.loads(request.content)["messages"]
        doc = message["content"].split("## CONTEXT:\n")[1].split("\n")[0]
        return httpx.Response(200, json=chat_body(*answers[doc]))

    entries = [entry(1, "a"), entry(0, "b")]  # sorted by document, as prepare_chat_entries does
    monkeypatch.setattr(rjb.judge_prompts, "load_documents_for_dataset", lambda *a: {"a": "", "b": ""})
    monkeypatch.setattr(rjb.judge_prompts, "prepare_chat_entries", lambda *a: entries)
    monkeypatch.setattr(rjb, "AsyncOpenAI", lambda **kw: AsyncOpenAI(
        **kw, max_retries=0, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))))
    cfg = {"model_id": "m", "max_concurrent": 2, "request_timeout": 5}
    monkeypatch.setattr(rjb.paths, "load_model_config", lambda *a: cfg)
    out = tmp_path / "out"

    def go(**params):
        rjb.run_judge_baseline(
            SimpleNamespace(name="pond", judge_instructions=None),
            "qwen-2.5-72b", VALID["verbalized"] | params, out, inp, seed=342, api_base="http://vllm.test/v1")
    return SimpleNamespace(go=go, out=out, answers=answers, entries=entries, inp=inp)


def test_e2e_scores_land_on_the_right_rows(e2e):
    e2e.go()
    got = json.loads((e2e.out / "responses.json").read_text())
    assert [(r["measurement_id"], r["judgement_p_true"]) for r in got] == [(10, 0.9), (11, 0.1)]
    assert "judgement" not in got[0] and not any(r["judgement_failed"] for r in got)
    # Scores are namespaced per judge when combined, and the judge key cannot vote.
    meta = json.loads((e2e.out / "run_metadata.json").read_text())
    assert meta["judge_model"] == "qwen-2.5-72b-verbalized"
    assert meta["confidence_counts"] == {"10": 1, "90": 1} and meta["n_rejected_answers"] == 0
    assert meta["method_params"] == {k: VALID["verbalized"][k] for k in rjb.METHOD_PARAMS["verbalized"]}
    assert meta["judge_model"] not in rjc.VOTING_JUDGE_KEYS
    (combined, _) = rjc.combine_judge_results(
        {meta["judge_model"]: e2e.out / "responses.json"}, set(), 1)
    assert combined["judgement_p_true_qwen-2.5-72b-verbalized"] == 0.9
    assert "judgement_p_true" not in combined and combined["judgement_combined"] is False


def test_e2e_failed_row_is_marked_and_the_run_completes(e2e):
    e2e.answers["b"] = ("unsure", "stop")
    e2e.go()
    failed, ok = json.loads((e2e.out / "responses.json").read_text())
    assert failed == {"measurement_id": 10, "document_id": "b", "judgement_p_true": None,
                      "judgement_raw_text": None, "judgement_model": None, "judgement_failed": True}
    assert ok["judgement_p_true"] == 0.1 and ok["judgement_failed"] is False
    raw = json.loads((e2e.out / "raw_outputs.json").read_text())
    assert raw["0"]["failed"] and raw["0"]["detail"] == {"rejected": ["unsure"] * 3}
    meta = json.loads((e2e.out / "run_metadata.json").read_text())
    assert (meta["n_rows"], meta["n_failed_rows"], meta["failed_row_indices"]) == (2, 1, [0])
    assert meta["confidence_counts"] == {"10": 1}


def test_e2e_every_row_failing_still_completes(e2e):
    e2e.answers.update(a=("unsure", "stop"), b=("unsure", "stop"))
    e2e.go()
    got = json.loads((e2e.out / "responses.json").read_text())
    assert all(r["judgement_failed"] and r["judgement_p_true"] is None for r in got)
    assert json.loads((e2e.out / "run_metadata.json").read_text())["max_prompt_tokens"] is None


def test_e2e_bug_is_not_disguised_as_a_failed_row(e2e, monkeypatch):
    def boom(text):
        raise KeyError("bug")
    monkeypatch.setattr(rjb, "parse_confidence", boom)
    with pytest.raises(KeyError):
        e2e.go()
    assert not (e2e.out / "responses.json").exists()
    raw = json.loads((e2e.out / "raw_outputs.json").read_text())
    assert any(r.get("aborted") and "KeyError" in r["error"] for r in raw.values())


def test_e2e_request_error_aborts_and_keeps_raw_outputs(e2e):
    e2e.answers["b"] = None  # handler fails -> connection error
    with pytest.raises(Exception, match="Connection error"):
        e2e.go()
    assert not (e2e.out / "responses.json").exists()
    assert "APIConnectionError" in json.loads((e2e.out / "raw_outputs.json").read_text())["0"]["error"]


def test_e2e_row_indices_select_rows_and_keep_their_input_index(e2e, monkeypatch):
    """Scoring only input row 1 uses row 1's seed and reports index 1."""
    seeds, real = [], rjb.sample_seed
    monkeypatch.setattr(rjb, "sample_seed", lambda *a: seeds.append(a) or real(*a))
    e2e.entries[:] = [e2e.entries[0] | {"custom_id": "0"}]  # document a, sole selected row
    e2e.go(row_indices=[1])
    (row,) = json.loads((e2e.out / "responses.json").read_text())
    assert (row["measurement_id"], row["judgement_p_true"]) == (11, 0.1)
    assert list(json.loads((e2e.out / "raw_outputs.json").read_text())) == ["1"]
    assert seeds == [(342, 1, 0, 0)]


def test_e2e_row_indices_out_of_range_raises(e2e):
    with pytest.raises(IndexError):
        e2e.go(row_indices=[2])


def test_e2e_dropped_prompt_row_raises(e2e):
    e2e.entries.pop()
    with pytest.raises(ValueError, match="1/2 rows have no judge prompt"):
        e2e.go()


def test_e2e_refuses_to_overwrite_or_rescore(e2e):
    e2e.go()
    with pytest.raises(FileExistsError):
        e2e.go()
    (e2e.out / "responses.json").unlink()
    e2e.inp.write_text(json.dumps([{"measurement_id": 1, "document_id": "a", "judgement_p_true": 0.2}]))
    with pytest.raises(ValueError, match="already carry"):
        e2e.go()
