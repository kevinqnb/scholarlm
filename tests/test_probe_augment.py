"""Rung-1 (model-free) unit tests for ``scholarlm.utils.probe_augment`` and the
``judge_common.prepare_chat_entries`` context-override plumbing.

Build note: ``notes/scholarlm/builds/2026-09-03-probe-synthetic-augmentation-01.md``.
Everything here runs against a hand-built fixture with the ``StubAugmentClient``
(no LLM), so the expected output is inspectable.
"""
from __future__ import annotations

import asyncio
import json
import random
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO / "experiments"))

from scholarlm.utils import probe_augment as pa


# ─── value / unit perturbation ────────────────────────────────────────────────


def test_perturb_value_preserves_precision_and_differs():
    rng = random.Random(0)
    for _ in range(200):
        out = pa.perturb_value("2.40", rng)
        assert out is not None
        assert out != "2.40"
        assert len(out.split(".")[1]) == 2          # 2 decimals kept
        assert 2.40 * 0.7 < float(out) < 2.40 * 1.3  # within ±25 %+slack


def test_perturb_value_integer_stays_integer():
    rng = random.Random(1)
    out = pa.perturb_value("100", rng)
    assert out is not None and "." not in out and out != "100"


def test_perturb_value_non_numeric_is_none():
    assert pa.perturb_value("acidic", random.Random(0)) is None
    assert pa.perturb_value("7.2 (mean)", random.Random(0)) is None


def test_alt_unit_family_bias_prefers_shared_token():
    rng = random.Random(0)
    units = ["µg/L", "mg/L", "ppb", "ppm", "mg/m^3"]
    picks = [pa.alt_unit("µg/L", units, rng, family_bias=1.0) for _ in range(50)]
    assert set(picks) == {"mg/L"}            # only unit sharing the "l" token
    assert pa.alt_unit("K", ["K"], rng) is None   # no alternative


# ─── edit application (fail loud) ─────────────────────────────────────────────


def test_apply_verified_edit_raises_on_missing_span():
    with pytest.raises(ValueError):
        pa.apply_verified_edit("the cat sat", [("dog", "cow")])


def test_apply_verified_edit_applies_present_span():
    assert pa.apply_verified_edit("the cat sat", [("cat", "bat")]) == "the bat sat"


# ─── stub client ─────────────────────────────────────────────────────────────


def test_stub_rewrite_applies_expected_and_flags_missing():
    stub = pa.StubAugmentClient()
    ok, ctx, reps, _ = stub.rewrite_context(
        context="Lake Bob has pH 7", instruction="x", expected=[("Bob", "Sue")]
    )
    assert ok and ctx == "Lake Sue has pH 7" and reps == [("Bob", "Sue")]
    ok, ctx, _, reason = stub.rewrite_context(
        context="Lake Bob has pH 7", instruction="x", expected=[("Zzz", "Sue")]
    )
    assert not ok and ctx == "Lake Bob has pH 7" and "not in context" in reason


# ─── GptOssClient.rewrite_context: the diff protocol ─────────────────────────
#
# Build note: 2026-09-XX diff-protocol section. The model returns ONLY a list of
# {"find","replace"} edits, each `find` a verbatim substring of the page; the
# client applies them itself (count=1 per edit) and verifies against the
# `expected` measurement spans. A missing `find` span is a skip (same category
# as `feasible: false`), not a crash (§4.1); a `feasible: true` with no usable
# `edits` list is a schema violation and crashes (§2.2 step 4).


def _rewrite_client(raw_response: str, monkeypatch, cache_path=None):
    """A GptOssClient whose `_one` returns `raw_response` verbatim (no server)."""
    async def _one(self, client, messages):
        return raw_response
    monkeypatch.setattr(pa.GptOssClient, "_one", _one)
    return pa.GptOssClient(api_base="x", cache=pa.AugmentCache(cache_path))


def test_rewrite_applies_model_edits_and_verifies(monkeypatch):
    raw = json.dumps({"feasible": True, "edits": [
        {"find": "Lake Bob has pH 7", "replace": "Lake Sue has pH 7"}]})
    c = _rewrite_client(raw, monkeypatch)
    ok, new_ctx, edits, reason = c.rewrite_context(
        context="Lake Bob has pH 7 today", instruction="i", expected=[("Bob", "Sue")])
    assert ok is True and reason == ""
    assert new_ctx == "Lake Sue has pH 7 today"
    assert edits == [("Lake Bob has pH 7", "Lake Sue has pH 7")]


def test_rewrite_patches_every_site_multi_edit(monkeypatch):
    # Same claim in prose, a table cell and a caption -> three edits, all applied.
    ctx = "The pond Bob was sampled. | Bob | 5 | \nFigure 2. Bob at dusk."
    raw = json.dumps({"feasible": True, "edits": [
        {"find": "The pond Bob was sampled.", "replace": "The pond Sue was sampled."},
        {"find": "| Bob | 5 |", "replace": "| Sue | 5 |"},
        {"find": "Figure 2. Bob at dusk.", "replace": "Figure 2. Sue at dusk."}]})
    c = _rewrite_client(raw, monkeypatch)
    ok, new_ctx, _, _ = c.rewrite_context(context=ctx, instruction="i",
                                          expected=[("Bob", "Sue")])
    assert ok is True
    assert "Bob" not in new_ctx and new_ctx.count("Sue") == 3


def test_rewrite_skips_when_find_span_absent(monkeypatch):
    raw = json.dumps({"feasible": True, "edits": [
        {"find": "a span that is not on the page", "replace": "x"}]})
    c = _rewrite_client(raw, monkeypatch)
    ok, new_ctx, edits, reason = c.rewrite_context(
        context="the actual page text", instruction="i", expected=[("page", "leaf")])
    assert ok is False and new_ctx == "the actual page text" and edits == []
    assert "not applicable" in reason


def test_rewrite_skips_when_later_edit_find_destroyed_by_earlier(monkeypatch):
    # edit #1's replacement removes edit #2's `find` span -> apply_verified_edit
    # raises mid-loop against the running text -> caught -> skip, not crash.
    ctx = "alpha beta gamma"
    raw = json.dumps({"feasible": True, "edits": [
        {"find": "alpha beta", "replace": "ALPHA"},
        {"find": "beta gamma", "replace": "GAMMA"}]})
    c = _rewrite_client(raw, monkeypatch)
    ok, new_ctx, edits, reason = c.rewrite_context(
        context=ctx, instruction="i", expected=[])
    assert ok is False and new_ctx == ctx and edits == [] and "not applicable" in reason


def test_rewrite_infeasible_is_a_clean_skip(monkeypatch):
    raw = json.dumps({"feasible": False, "reason": "would contradict Table 2"})
    c = _rewrite_client(raw, monkeypatch)
    ok, new_ctx, edits, reason = c.rewrite_context(
        context="ctx", instruction="i", expected=[("x", "y")])
    assert ok is False and new_ctx == "ctx" and edits == []
    assert reason == "would contradict Table 2"


def test_rewrite_feasible_but_no_edits_is_a_hard_error(monkeypatch):
    c = _rewrite_client(json.dumps({"feasible": True, "edits": []}), monkeypatch)
    with pytest.raises(ValueError, match="proposed no edits"):
        c.rewrite_context(context="ctx", instruction="i", expected=[])


@pytest.mark.parametrize("bad_edits", [
    "not a list",
    [{"find": "x"}],                       # missing replace
    [{"replace": "y"}],                    # missing find
    [{"find": 3, "replace": "y"}],         # find not a string
    ["justastring"],
])
def test_rewrite_malformed_edits_shape_is_a_hard_error(monkeypatch, bad_edits):
    c = _rewrite_client(json.dumps({"feasible": True, "edits": bad_edits}), monkeypatch)
    with pytest.raises(ValueError, match="edits.*not a list|edit is not a"):
        c.rewrite_context(context="ctx", instruction="i", expected=[])


def test_rewrite_repairs_commented_edits_then_applies(monkeypatch, capsys):
    raw = ('{"feasible": true, "edits": [\n'
           '  {"find": "in Bob Pond", "replace": "in Sue Pond"}  // rename\n'
           ']}')
    c = _rewrite_client(raw, monkeypatch)
    ok, new_ctx, _, _ = c.rewrite_context(
        context="chlorophyll rose in Bob Pond last year", instruction="i",
        expected=[("Bob Pond", "Sue Pond")])
    assert ok is True and new_ctx == "chlorophyll rose in Sue Pond last year"
    assert "line comment" in capsys.readouterr().out


def test_rewrite_repairs_unescaped_latex_in_find_then_applies(monkeypatch):
    # `\(`/`\)` copied verbatim from an OCR'd page span into `find`, unescaped.
    raw = '{"feasible": true, "edits": [{"find": "ratio \\( r = 0.5 \\)", "replace": "ratio \\( r = 0.9 \\)"}]}'
    c = _rewrite_client(raw, monkeypatch)
    ok, new_ctx, _, _ = c.rewrite_context(
        context="the ratio \\( r = 0.5 \\) was noted", instruction="i",
        expected=[("r = 0.5", "r = 0.9")])
    assert ok is True and new_ctx == "the ratio \\( r = 0.9 \\) was noted"


def test_rewrite_expected_check_still_catches_incomplete_edit(monkeypatch):
    # The GT span DOES appear verbatim and the model only covers one of two
    # sites -> `old in new_ctx` -> skip (the row's `valid` label would be wrong).
    ctx = "Site X value 5. Later, Site X value 5 again."
    raw = json.dumps({"feasible": True, "edits": [
        {"find": "Site X value 5.", "replace": "Site Y value 5."}]})
    c = _rewrite_client(raw, monkeypatch)
    ok, _, _, reason = c.rewrite_context(
        context=ctx, instruction="i", expected=[("Site X", "Site Y")])
    assert ok is False and "left the original span in place" in reason


def test_rewrite_fresh_miss_malformed_json_stays_out_of_cache(monkeypatch, tmp_path):
    # A malformed response on a live (non-record, non-strict) miss goes through
    # _run_batch, whose admission gate keeps it out of the cache and raises.
    raw = '{"feasible": true, "edits": [{"find": "a", "replace": "b"}, {"find": "c"'  # truncated
    c = _rewrite_client(raw, monkeypatch, cache_path=tmp_path / "cache.json")
    with pytest.raises(RuntimeError, match=r"1/1 gpt-oss call\(s\) failed"):
        c.rewrite_context(context="a c", instruction="i", expected=[])
    assert c.cache._store == {}


def test_rewrite_cache_key_carries_protocol_version(monkeypatch):
    # A cached OLD-protocol full-page response must NOT be read back by the new
    # parser: the payload's `protocol: 2` gives it a different key -> permanent
    # miss -> regenerated under the diff protocol.
    c = _rewrite_client(json.dumps({"feasible": True, "edits": []}), monkeypatch)
    old_key = pa._cache_key("rewrite", {"context": "ctx", "instruction": "i"})
    c.cache._store[old_key] = json.dumps(
        {"feasible": True, "context": "<whole page>", "replacements": []})
    # the new call computes a protocol-2 key, misses the stale entry, and (here)
    # raises on the canned empty-edits response rather than returning the page
    with pytest.raises(ValueError, match="proposed no edits"):
        c.rewrite_context(context="ctx", instruction="i", expected=[])


# ─── _extract_json_object: malformed gpt-oss responses ────────────────────────
#
# The generated text lands in the probe dataset, so a malformed gpt-oss
# response must fail loud, and the raw text must survive the failure for a
# post-mortem (it is not logged upstream and the cache flushes only at end of
# run). Two narrow, understood repairs are allowed, each tried only after a
# plain parse fails and each announced when it fires:
#   * unescaped backslashes copied from OCR'd LaTeX page spans (`\( n = 3 \)`);
#   * JS-style `// ...` line comments gpt-oss sometimes puts on a rewrite
#     response's `edits` array (first seen on the old-protocol `replacements`
#     array, nfix Rung-4, cached key bf3746a2...).
# Anything else (a token-repetition loop that runs out of budget mid-object,
# truncation) must keep failing loud. See the build note's Rung-4 entries.


def test_extract_json_object_dumps_raw_text_on_parse_failure(tmp_path):
    # Valid JSON up to a comma inside "edits", then a token that isn't a
    # legal value start -- an `Expecting value` failure, not a bad brace.
    bad = ('{"feasible": true, "edits": [{"find": "old", "replace": "new"}, ...]}')
    dump_dir = tmp_path / "bad_responses"

    with pytest.raises(ValueError, match="not valid JSON"):
        pa._extract_json_object(bad, op="rewrite", dump_dir=dump_dir)

    dumped = list(dump_dir.glob("bad_response_rewrite_*.txt"))
    assert len(dumped) == 1
    assert dumped[0].read_text() == bad


def test_extract_json_object_raises_without_dump_dir():
    # GptOssClient built on a path-less cache (unit tests, StubAugmentClient's
    # callers) has nowhere durable to dump -- must still fail loud, just
    # without a dump file.
    with pytest.raises(ValueError, match="not valid JSON"):
        pa._extract_json_object('{"a": [1, ...]}', op="event_fill", dump_dir=None)


def test_extract_json_object_valid_json_untouched():
    obj = pa._extract_json_object('{"feasible": false, "reason": "x"}',
                                  op="rewrite", dump_dir=None)
    assert obj == {"feasible": False, "reason": "x"}


def test_extract_json_object_valid_json_does_not_dump(tmp_path):
    # _dump_bad_response is called on three separate branches now; a regression
    # that dumps on the happy path would litter data/{ds}/bad_responses/ silently.
    dump_dir = tmp_path / "bad_responses"
    pa._extract_json_object('{"feasible": true, "edits": []}',
                            op="rewrite", dump_dir=dump_dir)
    assert not dump_dir.exists()


# The repair targets exactly one quirk: gpt-oss copies OCR'd LaTeX delimiters
# (`\( n = 3 \)`) verbatim into an `edits` `find` string with the backslash
# unescaped. Any other malformed response (e.g. a token-repetition loop that
# runs out of budget mid-object) must keep failing loud.


def test_extract_json_object_repairs_unescaped_latex_backslash():
    # A literal `\(`/`\)` pair copied into a `find` string unescaped.
    bad = ('{"feasible": true, "edits": [{"find": "Chlorophyll \\( a \\) rose", '
           '"replace": "Chlorophyll \\( b \\) rose"}]}')
    obj = pa._extract_json_object(bad, op="rewrite", dump_dir=None)
    assert obj["edits"][0]["find"] == "Chlorophyll \\( a \\) rose"


def test_extract_json_object_repair_leaves_valid_escapes_alone():
    # An already-correct `\\` escape (decoding to one literal backslash) must
    # not be miscounted as a second invalid backslash and doubled again -- a
    # real bug in the first version of the repair regex.
    bad = ('{"feasible": true, "edits": [{"find": "(\\\\( r = 0.5 \\\\))", '
           '"replace": "(\\\\( r = 0.9 \\\\))"}]}')
    obj = pa._extract_json_object(bad, op="rewrite", dump_dir=None)
    assert obj["edits"][0]["find"] == "(\\( r = 0.5 \\))"


def test_extract_json_object_repair_does_not_rescue_truncation_no_brace(tmp_path):
    # Ran out of budget mid-object, no closing brace at all -- caught before
    # the repair path even runs (start/end check). Still dumps and raises.
    bad = '{"feasible": true, "edits": [{"find": "a long verbatim span that got cut'  # no `}` at all
    dump_dir = tmp_path / "bad_responses"
    with pytest.raises(ValueError, match="not JSON"):
        pa._extract_json_object(bad, op="rewrite", dump_dir=dump_dir)
    assert len(list(dump_dir.glob("bad_response_rewrite_*.txt"))) == 1


def test_extract_json_object_repair_does_not_rescue_truncation_inner_brace(tmp_path):
    # Truncated mid-object, but `rfind("}")` lands on an inner object's brace,
    # so brace-trimming produces a still-unbalanced fragment. Reaches the repair
    # path, which can't help -- must dump and raise, not silently return the
    # half-object.
    bad = '{"feasible": true, "edits": [{"find": "b"}, {"replace":'
    dump_dir = tmp_path / "bad_responses"
    with pytest.raises(ValueError, match="not valid JSON"):
        pa._extract_json_object(bad, op="rewrite", dump_dir=dump_dir)
    assert len(list(dump_dir.glob("bad_response_rewrite_*.txt"))) == 1


# ── _strip_json_line_comments: gpt-oss annotates the `edits` array with `// ...` ──


def test_strip_json_line_comments_removes_comments_outside_strings():
    src = '{\n  "a": 1,  // trailing note\n  "b": [2, 3]  // another\n}'
    assert json.loads(pa._strip_json_line_comments(src)) == {"a": 1, "b": [2, 3]}


def test_strip_json_line_comments_preserves_double_slash_inside_a_string():
    # A `//` inside a value (a URL echoed from OCR'd page text) must survive --
    # the strip is string-aware, it is not a blind `//.*$` regex.
    src = '{"context": "see https://example.org/x for detail", "n": 1}'
    assert pa._strip_json_line_comments(src) == src
    assert json.loads(pa._strip_json_line_comments(src))["context"].endswith("/x for detail")


def test_strip_json_line_comments_noop_on_clean_json():
    src = '{"feasible": true, "edits": [{"find": "a / b", "replace": "c / d"}]}'
    assert pa._strip_json_line_comments(src) == src


def test_extract_json_object_repairs_js_line_comments_on_edits(capsys):
    # Modelled on the real nfix cached response bf3746a2... that crashed Rung-4
    # (there on the old `replacements` array): a well-formed rewrite object whose
    # `edits` array carries JS-style `// unchanged` annotations. Not legal JSON,
    # but the object must still parse so `feasible` / `edits` are usable.
    bad = (
        '{\n'
        '  "feasible": true,\n'
        '  "edits": [\n'
        '    {"find": "<td>S3</td>", "replace": "<td>Coastal Site Bravo</td>"},  // table cell\n'
        '    {"find": "sites S1-S3", "replace": "sites S1-Coastal Site Bravo"} // last one\n'
        '  ]\n'
        '}'
    )
    obj = pa._extract_json_object(bad, op="rewrite", dump_dir=None)
    assert obj["feasible"] is True
    assert len(obj["edits"]) == 2
    assert obj["edits"][0]["replace"] == "<td>Coastal Site Bravo</td>"
    assert "line comment" in capsys.readouterr().out   # fired loudly


def test_extract_json_object_does_not_dump_when_comment_strip_succeeds(tmp_path):
    bad = '{"feasible": false, "reason": "x"}  // no can do'
    dump_dir = tmp_path / "bad_responses"
    obj = pa._extract_json_object(bad, op="rewrite", dump_dir=dump_dir)
    assert obj == {"feasible": False, "reason": "x"}
    assert not dump_dir.exists()


def test_extract_json_object_combines_comment_strip_and_escape_repair(capsys):
    # One response with both quirks: a `//` comment AND an unescaped LaTeX
    # backslash in a `find` string. Needs the third (composed) entry in the ladder.
    bad = (
        '{"feasible": true, "edits": [\n'
        '  {"find": "Chlorophyll \\( a \\) rose", "replace": "Chlorophyll \\( b \\) rose"}  // trivial\n'
        ']}'
    )
    obj = pa._extract_json_object(bad, op="rewrite", dump_dir=None)
    assert obj["edits"][0]["find"] == "Chlorophyll \\( a \\) rose"
    assert "backslash" in capsys.readouterr().out


def test_extract_json_object_comment_strip_does_not_rescue_repetition_loop(tmp_path):
    # The other real nfix bad response (ced92306...): a token-repetition loop
    # that ran out of `max_tokens` mid-string. No `//` comments; stripping them
    # is a no-op and the unterminated string still fails. Must dump and raise.
    bad = ('{"feasible": true, "edits": ['
           + '{"find": "WCD", "replace": "ILBB"}, ' * 40 + '{"find": "WCD", "replace": "IL')
    dump_dir = tmp_path / "bad_responses"
    with pytest.raises(ValueError, match="not valid JSON|not JSON"):
        pa._extract_json_object(bad, op="rewrite", dump_dir=dump_dir)
    assert len(list(dump_dir.glob("bad_response_rewrite_*.txt"))) == 1


# ── _run_batch admission gate: never cache an unparseable response ────────────


def test_run_batch_keeps_unparseable_response_out_of_the_cache(tmp_path, monkeypatch):
    """A response `_one` returns but that will not parse as a JSON object (even
    after the repair ladder) must be treated as a batch failure and kept out of
    the cache -- otherwise `prewarm` flushes it as a "success" and it detonates
    in the strict-cache real pass with no way to regenerate just that key."""

    async def _one(self, client, messages):
        tag = messages[0]["content"]
        if tag == "loop":                       # unterminated string, no rescue
            return '{"feasible": true, "edits": [{"find": "a", "replace": "b"}, {"find": "a"'
        if tag == "commented":                  # repairable -> admitted (raw)
            return '{"feasible": false, "reason": "x"} // nope'
        return json.dumps({"feasible": True, "edits": []})

    monkeypatch.setattr(pa.GptOssClient, "_one", _one)
    cache_path = tmp_path / "cache.json"
    client = pa.GptOssClient(api_base="x", cache=pa.AugmentCache(cache_path))
    jobs = [("k_ok", [{"role": "user", "content": "ok"}]),
            ("k_loop", [{"role": "user", "content": "loop"}]),
            ("k_commented", [{"role": "user", "content": "commented"}])]

    with pytest.raises(RuntimeError, match=r"1/3 gpt-oss call\(s\) failed"):
        asyncio.run(client._run_batch(jobs))

    assert set(client.cache._store) == {"k_ok", "k_commented"}   # loop excluded
    assert client.cache._store["k_commented"].endswith("// nope")  # cached raw
    assert json.loads(cache_path.read_text()) == client.cache._store  # persisted


# ─── fixture ─────────────────────────────────────────────────────────────────


def _rules() -> pa.DatasetAugmentRules:
    return pa.DatasetAugmentRules(
        name="toy",
        entity_name_field="name",
        fabricated_names_by_type={"pond": ["Alpha Pond", "Beta Pond"],
                                  "lake": ["Gamma Lake"]},
        fabricated_names_any=["Fallback Water"],
        entity_type_token=lambda r: ("lake" if "lake" in (r.get("ecosystem") or "").lower()
                                     else "pond"),
        attr_units={"tn": ["µg/L", "mg/L", "ppb"], "tp": ["µg/L", "mg/L", "ppb"]},
        shared_unit_groups=[["tn", "tp"]],
        entity_field_locked=False,
        event_fields=["date"],
        event_prompt="EVENT FIELDS: date",
        event_synth_field="date",
        event_synth_value="Summer 2019",
        gt_cols=["document_id", "name", "ecosystem", "attribute", "value", "units",
                 "page_number", "date"],
    )


def _valid_record(i: int, name: str, eco: str, attr: str, val: str, units: str) -> dict:
    return {
        "document_id": f"D{i % 3}", "name": name, "ecosystem": eco,
        "attribute": attr, "value": val, "units": units, "page_number": 1, "date": None,
        "gt_row_index": i, "donor_gt_row_index": None,
        "_orig_idx": i, "_paper_code": f"D{i % 3}", "_page_numbers": [1],
        pa._CTX_ORIG_KEY: f"Site {name} reported {attr} of {val} {units} on page one.",
    }


def _fixture_valids(n: int) -> list[dict]:
    ecos = ["shallow pond", "karst lake"]
    return [_valid_record(i, f"Site {i}", ecos[i % 2],
                          "tn" if i % 2 else "tp", f"{10 + i}.0", "µg/L")
            for i in range(n)]


# ─── axis-2 positives ────────────────────────────────────────────────────────


def test_axis2_pos_entity_edits_context_and_measurement():
    rules, rng = _rules(), random.Random(0)
    stub = pa.StubAugmentClient()
    src = pa._prep_base_valid(_valid_record(0, "Site 0", "shallow pond", "tp", "10.0", "µg/L"), rules)
    row = pa.make_axis2_positive(src, "pos_entity", rules, stub, rng)
    assert row is not None
    assert row["label"] == "valid" and row["augment_axis"] == "pos_entity"
    assert row["name"] in {"Alpha Pond", "Beta Pond"}          # same-type pool
    assert row["name"] in row[pa._CTX_EDIT_KEY]                 # context edited
    assert "Site 0" not in row[pa._CTX_EDIT_KEY]
    assert row["source_group_id"] == 0
    assert "_is_base" not in row and "measurement_id" not in row


def test_axis2_pos_attribute_swaps_within_shared_unit_group():
    rules, rng = _rules(), random.Random(0)
    stub = pa.StubAugmentClient()
    src = pa._prep_base_valid(_valid_record(1, "Site 1", "karst lake", "tn", "11.0", "µg/L"), rules)
    row = pa.make_axis2_positive(src, "pos_attribute", rules, stub, rng)
    assert row is not None and row["attribute"] == "tp"


def test_axis2_pos_entity_locked_returns_none():
    rules = _rules()
    rules.entity_field_locked = True
    src = pa._prep_base_valid(_valid_record(0, "Site 0", "pond", "tp", "10.0", "µg/L"), rules)
    assert pa.make_axis2_positive(src, "pos_entity", rules, pa.StubAugmentClient(),
                                 random.Random(0)) is None


# ─── hard negatives ──────────────────────────────────────────────────────────


def test_hard_negative_on_edited_context_carries_the_edit():
    rules, rng = _rules(), random.Random(0)
    stub = pa.StubAugmentClient()
    src = pa._prep_base_valid(_valid_record(0, "Site 0", "shallow pond", "tp", "10.0", "µg/L"), rules)
    pos = pa.make_axis2_positive(src, "pos_entity", rules, stub, rng)
    neg = pa.make_hard_negative(pos, "hard_value", rules, rng, on_edited_context=True)
    assert neg["label"] == "invalid" and neg["modification_type"] == "hard_value"
    assert neg[pa._CTX_EDIT_KEY] == pos[pa._CTX_EDIT_KEY]        # same edited context
    assert neg["augment_axis"] == "pos_entity"


def test_hard_entity_skips_when_source_entity_field_is_null():
    # supermat case: sample_details is null for most rows -> a fabricated
    # "the paper says X" would be label noise, so hard_entity must skip it.
    rules, rng = _rules(), random.Random(0)
    src = pa._prep_base_valid(_valid_record(0, None, "pond", "tp", "10.0", "µg/L"), rules)
    assert src.get("name") is None
    assert pa.make_hard_negative(src, "hard_entity", rules, rng) is None
    # hard_value still fires — it corrupts a stated numeric claim
    assert pa.make_hard_negative(src, "hard_value", rules, rng) is not None


def test_gt_hard_negative_has_no_edited_context():
    rules, rng = _rules(), random.Random(0)
    src = pa._prep_base_valid(_valid_record(0, "Site 0", "pond", "tp", "10.0", "µg/L"), rules)
    neg = pa.make_hard_negative(src, "hard_value", rules, rng, on_edited_context=False)
    assert pa._CTX_EDIT_KEY not in neg
    assert neg["augment_axis"] is None


# ─── balance + per-source cap ────────────────────────────────────────────────


def test_balance_and_cap_exact_5050_and_derived_cap():
    rng = random.Random(0)
    rules = _rules()
    # one source, 1 base valid + 10 derived positives, 20 negatives
    base = pa._prep_base_valid(_valid_record(0, "Site 0", "pond", "tp", "10.0", "µg/L"), rules)
    derived_pos = [pa._new_derived_row(base, label="valid", mod_type=None, axis="pos_value")
                   for _ in range(10)]
    negs = [pa._new_derived_row(base, label="invalid", mod_type="hard_value", axis=None)
            for _ in range(20)]
    rows, report = pa.balance_and_cap([base] + derived_pos, negs, target=0, floor=0,
                                      max_derived_per_source=4, rng=rng)
    n_pos = sum(1 for r in rows if r["label"] == "valid")
    n_neg = len(rows) - n_pos
    assert n_pos == n_neg                       # exact 50/50
    # positives cap to 1 base + 4 derived = 5; negatives (all derived) cap to 4;
    # 50/50 then trims positives to match the minority -> 4/4.
    assert n_pos == 4
    assert report.n_capped_dropped == (10 - 4) + (20 - 4)
    # the one surviving base valid is never dropped by the cap
    assert any(r.get("_is_base") for r in rows)


def test_balance_and_cap_respects_target():
    rng = random.Random(0)
    rules = _rules()
    pos, neg = [], []
    for i in range(50):
        b = pa._prep_base_valid(_valid_record(i, f"S{i}", "pond", "tp", "1.0", "µg/L"), rules)
        pos.append(b)
        neg.append(pa._new_derived_row(b, label="invalid", mod_type="hard_value", axis=None))
    rows, report = pa.balance_and_cap(pos, neg, target=20, floor=10,
                                      max_derived_per_source=4, rng=rng)
    assert len(rows) == 20 and report.hit_target and report.hit_floor


# ─── context-override inlining + full pipeline determinism ───────────────────


def test_inline_context_overrides_only_edited_rows():
    rows = [
        {"measurement_id": 0, pa._CTX_ORIG_KEY: "a", pa._CTX_EDIT_KEY: "A"},
        {"measurement_id": 1, pa._CTX_ORIG_KEY: "b"},                       # unedited
        {"measurement_id": 2, pa._CTX_ORIG_KEY: "c", pa._CTX_EDIT_KEY: "c"},  # noop edit
    ]
    n = pa.inline_context_overrides(rows)
    assert n == 1
    assert rows[0]["context_override"] == "A"
    assert "context_override" not in rows[1]
    assert "context_override" not in rows[2]


def _run_pipeline(seed: int, tmp_path: Path) -> dict:
    rules = pa._rules_for_test = _rules()
    flags = pa.AugmentFlags(augment_events=False, pos_axes=("pos_entity", "pos_attribute"),
                            target_rows=0, floor_rows=0, max_derived_per_source=4)
    valids = _fixture_valids(24)
    tr, te = valids[:16], valids[16:]
    return pa.build_augmented_files(
        xv_train=[dict(v) for v in tr], xv_test=[dict(v) for v in te],
        rng=random.Random(seed), client=pa.StubAugmentClient(), rules=rules, flags=flags,
    )


def test_pipeline_is_seed_deterministic(tmp_path):
    a = _run_pipeline(42, tmp_path)
    b = _run_pipeline(42, tmp_path)
    for key in ("train", "primary_test", "diagnostic_test"):
        assert a[key][0] == b[key][0], f"{key} rows differ across two seed-42 runs"
        assert a[key][1] == b[key][1], f"{key} raw (pre-strip) rows differ across two seed-42 runs"


def test_pipeline_files_are_wellformed(tmp_path):
    out = _run_pipeline(42, tmp_path)
    for key in ("train", "primary_test", "diagnostic_test"):
        rows, _ = out[key]
        pa.assert_wellformed(key, rows)
        assert all("source_group_id" in r for r in rows)
    # diagnostic file: every row sits on an edited context, inlined on the row.
    # Guard against the vacuous pass — an empty diagnostic split (every axis-2
    # rewrite skipped) would satisfy `all(...)` trivially.
    d_rows, _ = out["diagnostic_test"]
    assert d_rows, "no diagnostic-test rows — every axis-2 positive was skipped"
    assert all(r.get("context_override") for r in d_rows)


def test_run_and_write_inlines_overrides_no_sidecar_file(tmp_path):
    """run_and_write -> data files carry context_override inline; no side-car
    file is written; judge_common.prepare_chat_entries picks the field straight
    off the row, per-row-aligned (replaces the retired side-car round-trip)."""
    import judge_common
    rules = _rules()
    flags = pa.AugmentFlags(augment_events=False, pos_axes=("pos_entity", "pos_attribute"),
                            target_rows=0, floor_rows=0, max_derived_per_source=4)
    valids = _fixture_valids(24)
    # OCR files: one per _paper_code, each page-1 block naming its sites verbatim
    # so the stub rewrite can find the entity span.
    ocr_dir = tmp_path / "ocr"
    ocr_dir.mkdir()
    by_code: dict[str, list[dict]] = {}
    for v in valids:
        by_code.setdefault(v["_paper_code"], []).append(v)
    for code, vs in by_code.items():
        body = " ".join(f"{v['name']} reported {v['attribute']} of {v['value']} {v['units']}."
                        for v in vs)
        (ocr_dir / f"{code}.txt").write_text(f'<page number="1">{body}</page>')

    written = pa.run_and_write(
        base_dir=tmp_path, out_suffix="_v2", ocr_dir=ocr_dir,
        xv_train=[dict(v) for v in valids[:16]], xv_test=[dict(v) for v in valids[16:]],
        rules=rules, flags=flags, client=pa.StubAugmentClient(), rng=random.Random(42),
    )
    train_p, primary_p, diag_p = (written["train"], written["primary_test"],
                                  written["diagnostic_test"])

    # no side-car files anywhere in the output directory
    assert not list(tmp_path.glob("probe_context_overrides*"))
    # a diff report exists for the two splits that have edited rows, not primary
    assert (tmp_path / (train_p.name + ".diff.txt")).exists()
    assert (tmp_path / (diag_p.name + ".diff.txt")).exists()
    assert not (tmp_path / (primary_p.name + ".diff.txt")).exists()

    train_data = json.loads(train_p.read_text())
    diag_data = json.loads(diag_p.read_text())
    primary_data = json.loads(primary_p.read_text())
    n_train_edited = sum(1 for r in train_data if r.get("context_override"))
    assert n_train_edited > 0 and n_train_edited < len(train_data)
    assert all(r.get("context_override") for r in diag_data)         # fully synthetic
    assert not any(r.get("context_override") for r in primary_data)  # never edited

    cfg = _dcfg()
    documents = {v["_paper_code"]: (ocr_dir / f"{v['_paper_code']}.txt").read_text()
                for v in valids}

    for data, has_any_edits in [(train_data, True), (diag_data, True), (primary_data, False)]:
        entries = judge_common.prepare_chat_entries(data, documents, cfg)
        for entry in entries:
            orig_idx = int(entry["custom_id"])
            row = data[orig_idx]
            override = row.get("context_override")
            if override is not None:
                assert entry["page_text"] == override
            else:
                pn = row.get("page_number")
                page_numbers = ([p for p in pn if p is not None] if isinstance(pn, list)
                                else ([pn] if pn is not None else []))
                assert entry["page_text"] == judge_common.extract_page_text(
                    documents[str(row["document_id"])], page_numbers)
        if has_any_edits:
            assert any(e["page_text"] == data[int(e["custom_id"])].get("context_override")
                      for e in entries if data[int(e["custom_id"])].get("context_override"))


# ─── GptOssClient prewarm / record-pass plumbing ─────────────────────────────
#
# A full generation run does a `record` dry pass through `build_augmented_files`
# to collect every gpt-oss prompt, `prewarm`s them in one concurrent batch, then
# runs for real against a warm cache.  These tests use a canned `_run_batch`
# (no server) to check: the record pass calls no model and leaves the cache
# untouched; two prewarm orderings produce a byte-identical cache; and the real
# pass that follows a record pass makes zero model calls (RNG parity — the whole
# point of the phase reorder in `build_augmented_files`).


def _canned_run_batch(*, event_date: str | None = None):
    """An async `_run_batch` stand-in: stub-style diff-protocol edits, canned
    event-fill.

    Caches each result itself -- the real `_run_batch` owns that (so a
    partial-batch failure doesn't discard sibling successes), and this stands
    in for the whole method, so it must honor the same contract.
    """

    async def _run_batch(self, jobs):
        out: dict[str, str] = {}
        for key, messages in jobs:
            user = messages[-1]["content"]
            if "## MEASUREMENT\n" in user:                       # event_fill
                out[key] = json.dumps({} if event_date is None else {"date": event_date})
                continue
            instr, page = user.split("## PAGE TEXT\n", 1)        # rewrite (diff protocol)
            quoted = re.findall(r'"([^"]+)"', instr)
            olds = [t for t in quoted if t in page]              # spans verbatim in the page
            news = [t for t in quoted if t not in page]          # candidate replacements
            if not olds:
                out[key] = json.dumps(
                    {"feasible": False, "reason": "canned: no page-verbatim span"})
                continue
            edits = [{"find": o, "replace": (news[0] if news else o + "_X")} for o in olds]
            out[key] = json.dumps({"feasible": True, "edits": edits})
        for key, raw in out.items():
            self.cache.put(key, raw)
        return out

    return _run_batch


def _augment_fixture(axes=("pos_entity", "pos_attribute"), *, events=False):
    rules = _rules()
    flags = pa.AugmentFlags(augment_events=events, pos_axes=axes,
                            target_rows=0, floor_rows=0, max_derived_per_source=4)
    valids = _fixture_valids(24)
    tr = [dict(v) for v in valids[:16]]
    te = [dict(v) for v in valids[16:]]
    return tr, te, rules, flags


def _record_pass(client, tr, te, rules, flags, seed=42):
    """Run the `record` dry pass; return (jobs, rng) with rng rewound to pre-pass."""
    rng = random.Random(seed)
    state = rng.getstate()
    client.record = True
    pa.build_augmented_files(xv_train=tr, xv_test=te, rng=rng,
                             client=client, rules=rules, flags=flags, quiet=True)
    client.record = False
    jobs = list(client._pending.items())
    client._pending.clear()
    rng.setstate(state)
    return jobs, rng


def test_gptoss_record_pass_collects_prompts_and_leaves_cache_clean(monkeypatch):
    async def _boom(self, jobs):
        raise AssertionError("the record pass must not call the model")

    monkeypatch.setattr(pa.GptOssClient, "_run_batch", _boom)
    client = pa.GptOssClient(api_base="http://unused/v1", cache=pa.AugmentCache(None))
    jobs, _ = _record_pass(client, *_augment_fixture())

    assert len(jobs) > 0
    assert all(len(k) == 64 and re.fullmatch(r"[0-9a-f]{64}", k) for k, _ in jobs)
    # the record pass never writes the cache — a canned `feasible: false` landing
    # in the store would poison every subsequent run
    assert client.cache._store == {}
    assert client.cache._misses == 0


# A fake AsyncOpenAI `client.chat.completions.create` for testing `_one`
# directly, without a real server. gpt-oss's harmony chat template has no
# `enable_thinking` variable (checked against the cached chat_template.jinja),
# only `reasoning_effort`, so `_one` must send that via `extra_body`.
class _FakeChoice:
    def __init__(self, content, finish_reason="stop", reasoning_content=None):
        self.finish_reason = finish_reason
        self.message = SimpleNamespace(content=content, reasoning_content=reasoning_content)


class _FakeChatCompletions:
    def __init__(self, resp, captured):
        self._resp, self._captured = resp, captured

    async def create(self, **kwargs):
        self._captured.update(kwargs)
        return self._resp


class _FakeClient:
    def __init__(self, choice, captured):
        self.chat = SimpleNamespace(
            completions=_FakeChatCompletions(SimpleNamespace(choices=[choice]), captured))


def test_one_sends_reasoning_effort():
    client = pa.GptOssClient(api_base="x", cache=pa.AugmentCache(None), reasoning_effort="low")
    captured: dict = {}
    fake = _FakeClient(_FakeChoice("hello"), captured)

    out = asyncio.run(client._one(fake, [{"role": "user", "content": "hi"}]))

    assert out == "hello"
    assert captured["extra_body"] == {"chat_template_kwargs": {"reasoning_effort": "low"}}


def test_one_empty_content_error_carries_finish_reason_and_reasoning_length():
    client = pa.GptOssClient(api_base="x", cache=pa.AugmentCache(None))
    fake = _FakeClient(_FakeChoice(None, finish_reason="length", reasoning_content="x" * 500), {})

    with pytest.raises(ValueError, match=r"finish_reason='length'.*reasoning_content_len=500"):
        asyncio.run(client._one(fake, [{"role": "user", "content": "hi"}]))


def test_one_raises_on_length_finish_with_nonempty_content():
    # Truncated at max_tokens but content non-empty: caught at the call site with
    # the diagnostic fields attached, not two layers down as an opaque parse
    # failure. Under the diff protocol the output is tiny so this should not
    # fire, but truncation is exactly what the protocol change set out to kill.
    client = pa.GptOssClient(api_base="x", cache=pa.AugmentCache(None))
    fake = _FakeClient(_FakeChoice('{"feasible": tr', finish_reason="length"), {})

    with pytest.raises(ValueError, match=r"truncated at max_tokens.*finish_reason='length'"):
        asyncio.run(client._one(fake, [{"role": "user", "content": "hi"}]))


def test_run_batch_partial_failure_caches_successes_before_raising(tmp_path, monkeypatch):
    """One job in a `prewarm` batch fails. `asyncio.gather` without
    `return_exceptions=True` would propagate that and discard every sibling
    response already computed in the same batch. `_run_batch` must cache and
    persist the successes before it re-raises."""

    async def _one(self, client, messages):
        if messages[0]["content"] == "fail":
            raise ValueError("simulated gpt-oss failure")
        return json.dumps({"feasible": True, "edits": []})

    monkeypatch.setattr(pa.GptOssClient, "_one", _one)
    cache_path = tmp_path / "cache.json"
    client = pa.GptOssClient(api_base="x", cache=pa.AugmentCache(cache_path))
    jobs = [(f"k{i}", [{"role": "user", "content": "fail" if i == 1 else "ok"}])
            for i in range(4)]

    with pytest.raises(RuntimeError, match=r"1/4 gpt-oss call\(s\) failed"):
        asyncio.run(client._run_batch(jobs))

    assert set(client.cache._store) == {"k0", "k2", "k3"}   # the 3 successes
    assert cache_path.exists()                                # persisted, not just in memory
    assert json.loads(cache_path.read_text()) == client.cache._store


def test_prewarm_is_order_independent(tmp_path, monkeypatch):
    monkeypatch.setattr(pa.GptOssClient, "_run_batch", _canned_run_batch())
    p1, p2 = tmp_path / "c1.json", tmp_path / "c2.json"

    c1 = pa.GptOssClient(api_base="x", cache=pa.AugmentCache(p1))
    jobs, _ = _record_pass(c1, *_augment_fixture())
    assert len(jobs) >= 3
    c1.prewarm(jobs)

    c2 = pa.GptOssClient(api_base="x", cache=pa.AugmentCache(p2))
    c2.prewarm(list(reversed(jobs)))

    assert c1.cache._store == c2.cache._store
    c1.cache.save()
    c2.cache.save()
    assert p1.read_bytes() == p2.read_bytes()      # committed cache: no ordering diff


def test_prewarm_cache_survives_a_crash_in_the_post_prewarm_assembly(tmp_path, monkeypatch):
    """`run_and_write` must persist prewarm's model-generated responses before
    running the real pass, so a crash anywhere after prewarm (including inside
    build_augmented_files itself) costs a cache reload, not a re-run against
    the model."""
    monkeypatch.setattr(pa.GptOssClient, "_run_batch", _canned_run_batch())
    rules = _rules()
    flags = pa.AugmentFlags(augment_events=False, pos_axes=("pos_entity", "pos_attribute"),
                            target_rows=0, floor_rows=0, max_derived_per_source=4)
    valids = _fixture_valids(24)
    ocr_dir = tmp_path / "ocr"
    ocr_dir.mkdir()
    by_code: dict[str, list[dict]] = {}
    for v in valids:
        by_code.setdefault(v["_paper_code"], []).append(v)
    for code, vs in by_code.items():
        body = " ".join(f"{v['name']} reported {v['attribute']} of {v['value']} {v['units']}."
                        for v in vs)
        (ocr_dir / f"{code}.txt").write_text(f'<page number="1">{body}</page>')

    cache_path = tmp_path / "cache.json"
    client = pa.GptOssClient(api_base="x", cache=pa.AugmentCache(cache_path))

    real_build = pa.build_augmented_files

    def _boom_after_prewarm(*, client, **kwargs):
        if not client.record:      # the real pass: prewarm has already run
            raise RuntimeError("simulated crash in assembly, after prewarm")
        return real_build(client=client, **kwargs)

    monkeypatch.setattr(pa, "build_augmented_files", _boom_after_prewarm)

    with pytest.raises(RuntimeError, match="simulated crash"):
        pa.run_and_write(
            base_dir=tmp_path, out_suffix="_v2", ocr_dir=ocr_dir,
            xv_train=[dict(v) for v in valids[:16]], xv_test=[dict(v) for v in valids[16:]],
            rules=rules, flags=flags, client=client, rng=random.Random(42),
        )

    assert cache_path.exists(), "prewarm's responses must survive a crash later in the same run"
    saved = json.loads(cache_path.read_text())
    assert saved == client.cache._store
    assert len(saved) > 0


@pytest.mark.parametrize("resumed", [False, True])
def test_record_then_real_pass_makes_zero_model_calls(monkeypatch, resumed):
    monkeypatch.setattr(pa.GptOssClient, "_run_batch", _canned_run_batch())
    tr, te, rules, flags = _augment_fixture()

    # discover the rewrite keys with a throwaway probe client
    probe = pa.GptOssClient(api_base="x", cache=pa.AugmentCache(None))
    discovered, _ = _record_pass(probe, tr, te, rules, flags)
    rewrite_keys = [k for k, m in discovered if "## PAGE TEXT\n" in m[-1]["content"]]
    assert rewrite_keys

    client = pa.GptOssClient(api_base="x", cache=pa.AugmentCache(None))
    if resumed:
        # a crashed-and-restarted run: some rewrites are already cached.  The
        # cached value need not verify — RNG draws in `_axis2_positives` happen
        # before the client call and do not depend on its return. A cached
        # `feasible: false` just means that axis-2 positive is skipped.
        for k in rewrite_keys[::2]:
            client.cache._store[k] = json.dumps(
                {"feasible": False, "reason": "stale (resumed-run cache)"})

    jobs, rng = _record_pass(client, tr, te, rules, flags)
    client.prewarm(jobs)
    misses_after_prewarm = client.cache._misses
    client.strict_cache = True

    out = pa.build_augmented_files(xv_train=tr, xv_test=te, rng=rng,
                                   client=client, rules=rules, flags=flags)

    # strict_cache did not raise, and not one new model call happened
    assert client.cache._misses == misses_after_prewarm
    for key in ("train", "primary_test", "diagnostic_test"):
        pa.assert_wellformed(key, out[key][0])
    # the canned `_run_batch` produced real edits (find spans are in the fixture
    # pages) — the diagnostic split is non-empty, not a vacuous pass
    diag_rows = out["diagnostic_test"][0]
    assert diag_rows and all(r.get("context_override") for r in diag_rows)


def test_strict_cache_raises_when_eventfill_result_feeds_pos_event(monkeypatch):
    # Known residual dependency: event-fill output changes the pos_event rewrite
    # instruction (`if cur:` vs the inject-a-sentence branch), so with
    # `--augment-pos-axes pos_event` the record pass (canned empty event-fill)
    # collects the wrong rewrite prompt.  strict_cache must catch this loudly.
    monkeypatch.setattr(pa.GptOssClient, "_run_batch", _canned_run_batch(event_date="2019"))
    tr, te, rules, flags = _augment_fixture(axes=("pos_entity", "pos_event"), events=True)

    client = pa.GptOssClient(api_base="x", cache=pa.AugmentCache(None))
    jobs, rng = _record_pass(client, tr, te, rules, flags)
    client.prewarm(jobs)
    client.strict_cache = True

    with pytest.raises(RuntimeError, match="pos_event"):
        pa.build_augmented_files(xv_train=tr, xv_test=te, rng=rng,
                                 client=client, rules=rules, flags=flags)


# ─── judge_common.prepare_chat_entries context-override plumbing ──────────────


def _dcfg():
    from run_extraction import load_dataset_config
    return load_dataset_config("pond")


def test_prepare_chat_entries_no_override_field_is_unaffected():
    """A row with no context_override key behaves exactly like every row did
    before this field existed — byte-identical prompt from the OCR lookup."""
    import judge_common
    cfg = _dcfg()
    data = [{"document_id": "X", "attribute": "tp", "value": "5", "units": "µg/L",
             "measurement_id": 0, "name": "L", "page_number": [1]}]
    docs = {"X": '<page number="1">total phosphorus 5 µg/L</page>'}
    entries = judge_common.prepare_chat_entries(data, docs, cfg)
    assert entries[0]["page_text"] == judge_common.extract_page_text(docs["X"], [1])


def test_prepare_chat_entries_null_override_is_inert():
    """context_override: null (the JSON round-trip of an absent field) behaves
    the same as the field being absent entirely."""
    import judge_common
    cfg = _dcfg()
    base = {"document_id": "X", "attribute": "tp", "value": "5", "units": "µg/L",
            "measurement_id": 0, "name": "L", "page_number": [1]}
    docs = {"X": '<page number="1">total phosphorus 5 µg/L</page>'}
    a = judge_common.prepare_chat_entries([dict(base)], docs, cfg)
    b = judge_common.prepare_chat_entries([dict(base, context_override=None)], docs, cfg)
    assert a == b


def test_prepare_chat_entries_applies_row_override():
    import judge_common
    cfg = _dcfg()
    data = [{"document_id": "X", "attribute": "tp", "value": "5", "units": "µg/L",
             "measurement_id": 7, "name": "L", "page_number": [1],
             "context_override": "REWRITTEN CONTEXT for the probe"}]
    docs = {"X": '<page number="1">total phosphorus 5 µg/L</page>'}
    entries = judge_common.prepare_chat_entries(data, docs, cfg)
    assert entries[0]["page_text"] == "REWRITTEN CONTEXT for the probe"
    assert "REWRITTEN CONTEXT" in entries[0]["user"]


def test_prepare_chat_entries_override_is_per_row():
    """One row's override does not leak onto a sibling row with none."""
    import judge_common
    cfg = _dcfg()
    data = [
        {"document_id": "X", "attribute": "tp", "value": "5", "units": "µg/L",
         "measurement_id": 0, "name": "L", "page_number": [1],
         "context_override": "EDITED"},
        {"document_id": "X", "attribute": "tp", "value": "6", "units": "µg/L",
         "measurement_id": 1, "name": "M", "page_number": [1]},
    ]
    docs = {"X": '<page number="1">total phosphorus 5 µg/L</page>'}
    entries = judge_common.prepare_chat_entries(data, docs, cfg)
    by_mid = {int(e["custom_id"]): e for e in entries}
    assert by_mid[0]["page_text"] == "EDITED"
    assert by_mid[1]["page_text"] == judge_common.extract_page_text(docs["X"], [1])


# ─── --synthetic-name path routing (experiments/paths.py) ─────────────────────


def test_synthetic_probe_named_routes_to_dedicated_tree():
    import paths
    p = paths.synthetic_probe_named("pond", "v2_diag", "mistral-7b", "2026_09_10")
    assert p.parts[-3:] == ("synthetic_probe_v2_diag", "mistral-7b", "2026_09_10")


def test_synthetic_probe_split_vs_name_precedence():
    import paths
    assert paths.synthetic_probe("pond", "m").parts[-3] == "synthetic_probe"
    assert paths.synthetic_probe_test("pond", "m").parts[-3] == "synthetic_probe_test"
    # name wins over the default split
    assert paths.synthetic_probe("pond", "m", name="foo").parts[-3] == "synthetic_probe_foo"


def test_synthetic_name_rejects_bad_chars():
    import paths
    for bad in ("Bad-Name", "has space", "UPPER", "trailing/slash", ""):
        with pytest.raises(ValueError):
            paths.synthetic_probe_named("pond", bad, "m")


def test_find_synthetic_carries_name_through(tmp_path, monkeypatch):
    import paths
    monkeypatch.setattr(paths, "EXPERIMENTS_ROOT", tmp_path)
    run = tmp_path / "pond" / "synthetic_probe_v2_diag" / "mistral-7b" / "2026_09_10"
    run.mkdir(parents=True)
    (run / "responses.json").write_text("[]")
    got = paths.find_synthetic_responses("pond", "mistral-7b", "2026_09_10", name="v2_diag")
    assert got == run / "responses.json"
    # same lookup without the name looks in the wrong tree and fails loud
    with pytest.raises(FileNotFoundError):
        paths.find_synthetic_responses("pond", "mistral-7b", "2026_09_10")

