"""Rung-1 (model-free) unit tests for ``scholarlm.utils.probe_augment`` and the
``judge_common.prepare_chat_entries`` context-override plumbing.

Build note: ``notes/scholarlm/builds/2026-09-03-probe-synthetic-augmentation-01.md``.
Everything here runs against a hand-built fixture with the ``StubAugmentClient``
(no LLM) or a monkeypatched ``_one`` / ``_run_batch``, so the expected output is
inspectable.
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
        assert len(out.split(".")[1]) == 2
        assert 2.40 * 0.7 < float(out) < 2.40 * 1.3


def test_perturb_value_integer_stays_integer():
    out = pa.perturb_value("100", random.Random(1))
    assert out is not None and "." not in out and out != "100"


def test_perturb_value_non_numeric_is_none():
    assert pa.perturb_value("acidic", random.Random(0)) is None
    assert pa.perturb_value("7.2 (mean)", random.Random(0)) is None


# ─── edit application (fail loud) ─────────────────────────────────────────────


def test_apply_verified_edit_raises_on_missing_span():
    with pytest.raises(ValueError):
        pa.apply_verified_edit("the cat sat", [("dog", "cow")])


def test_apply_verified_edit_applies_present_span():
    assert pa.apply_verified_edit("the cat sat", [("cat", "bat")]) == "the bat sat"


# ─── GptOssClient.rewrite_context: protocol 3 ────────────────────────────────
#
# The model returns {"feasible": true, "replacement": "<new value>", "edits":
# [{"find","replace"}]}. The client applies the edits (count=1 per edit),
# verifies the replacement landed and the original surface value is gone where
# it was present, and reports `unverified_span` when `original` was never a
# verbatim page substring. A missing `find` span, an infeasible verdict, or a
# failed verification are all clean skips; a feasible verdict with no
# `replacement` string or no usable `edits` list crashes.


def _rewrite_client(raw_response, monkeypatch, cache_path=None):
    async def _one(self, client, messages):
        return raw_response
    monkeypatch.setattr(pa.GptOssClient, "_one", _one)
    return pa.GptOssClient(api_base="x", cache=pa.AugmentCache(cache_path))


def test_rewrite_applies_model_edits_and_verifies(monkeypatch):
    raw = json.dumps({"feasible": True, "replacement": "Sue Pond", "edits": [
        {"find": "Bob Pond has pH 7", "replace": "Sue Pond has pH 7"}]})
    c = _rewrite_client(raw, monkeypatch)
    res = c.rewrite_context(context="Bob Pond has pH 7 today", instruction="i",
                            original="Bob Pond")
    assert res.applied is True and res.reason == ""
    assert res.new_context == "Sue Pond has pH 7 today"
    assert res.replacement == "Sue Pond"
    assert res.edits == [("Bob Pond has pH 7", "Sue Pond has pH 7")]
    assert res.unverified_span is False


def test_rewrite_patches_every_site_multi_edit(monkeypatch):
    ctx = "The pond Bob was sampled. | Bob | 5 |\nFigure 2. Bob at dusk."
    raw = json.dumps({"feasible": True, "replacement": "Sue", "edits": [
        {"find": "The pond Bob was sampled.", "replace": "The pond Sue was sampled."},
        {"find": "| Bob | 5 |", "replace": "| Sue | 5 |"},
        {"find": "Figure 2. Bob at dusk.", "replace": "Figure 2. Sue at dusk."}]})
    c = _rewrite_client(raw, monkeypatch)
    res = c.rewrite_context(context=ctx, instruction="i", original="Bob")
    assert res.applied and "Bob" not in res.new_context and res.new_context.count("Sue") == 3


def test_rewrite_skips_when_find_span_absent(monkeypatch):
    raw = json.dumps({"feasible": True, "replacement": "x", "edits": [
        {"find": "a span that is not on the page", "replace": "x"}]})
    c = _rewrite_client(raw, monkeypatch)
    res = c.rewrite_context(context="the actual page text", instruction="i",
                            original="page")
    assert res.applied is False and res.new_context == "the actual page text"
    assert "not applicable" in res.reason


def test_rewrite_skips_when_later_edit_find_destroyed_by_earlier(monkeypatch):
    raw = json.dumps({"feasible": True, "replacement": "X", "edits": [
        {"find": "alpha beta", "replace": "ALPHA"},
        {"find": "beta gamma", "replace": "GAMMA"}]})
    c = _rewrite_client(raw, monkeypatch)
    res = c.rewrite_context(context="alpha beta gamma", instruction="i", original="alpha")
    assert res.applied is False and "not applicable" in res.reason


def test_rewrite_infeasible_is_a_clean_skip(monkeypatch):
    c = _rewrite_client(json.dumps({"feasible": False, "reason": "would contradict Table 2"}),
                        monkeypatch)
    res = c.rewrite_context(context="ctx", instruction="i", original="x")
    assert res.applied is False and res.new_context == "ctx"
    assert res.reason == "would contradict Table 2"


def test_rewrite_feasible_but_no_replacement_is_a_hard_error(monkeypatch):
    c = _rewrite_client(json.dumps({"feasible": True, "edits": [{"find": "a", "replace": "b"}]}),
                        monkeypatch)
    with pytest.raises(ValueError, match="replacement.*missing/empty"):
        c.rewrite_context(context="a", instruction="i", original="a")


def test_rewrite_feasible_but_no_edits_is_a_hard_error(monkeypatch):
    c = _rewrite_client(json.dumps({"feasible": True, "replacement": "b"}), monkeypatch)
    with pytest.raises(ValueError, match="proposed no edits"):
        c.rewrite_context(context="ctx", instruction="i", original="c")


@pytest.mark.parametrize("bad_edits", [
    "not a list", [{"find": "x"}], [{"replace": "y"}], [{"find": 3, "replace": "y"}], ["str"],
])
def test_rewrite_malformed_edits_shape_is_a_hard_error(monkeypatch, bad_edits):
    c = _rewrite_client(json.dumps({"feasible": True, "replacement": "z", "edits": bad_edits}),
                        monkeypatch)
    with pytest.raises(ValueError, match="edits.*not a list|edit is not a"):
        c.rewrite_context(context="ctx", instruction="i", original="c")


def test_rewrite_skips_when_original_left_on_page(monkeypatch):
    # The GT span appears verbatim twice and the model only covers one -> skip
    # (the row's `valid` label would be wrong).
    ctx = "Site X value 5. Later, Site X value 5 again."
    raw = json.dumps({"feasible": True, "replacement": "Site Y", "edits": [
        {"find": "Site X value 5.", "replace": "Site Y value 5."}]})
    c = _rewrite_client(raw, monkeypatch)
    res = c.rewrite_context(context=ctx, instruction="i", original="Site X")
    assert res.applied is False and "left the original" in res.reason


def test_rewrite_flags_unverified_span_when_original_absent(monkeypatch):
    raw = json.dumps({"feasible": True, "replacement": "42.0", "edits": [
        {"find": "a value of forty", "replace": "a value of 42.0"}]})
    c = _rewrite_client(raw, monkeypatch)
    res = c.rewrite_context(context="here we have a value of forty today",
                            instruction="i", original="40")
    assert res.applied is True and res.unverified_span is True


def test_rewrite_inject_case_requires_a_new_span(monkeypatch):
    raw = json.dumps({"feasible": True, "replacement": "Summer 2019", "edits": [
        {"find": "value 5 here", "replace": "value 5 here (Summer 2019)"}]})
    c = _rewrite_client(raw, monkeypatch)
    res = c.rewrite_context(context="value 5 here", instruction="i", original="")
    assert res.applied is True and res.unverified_span is True
    res2 = c.rewrite_context(context="value 5 here Summer 2019", instruction="i", original="")
    assert res2.applied is False and "already in the paper" in res2.reason


def test_rewrite_repairs_commented_edits_then_applies(monkeypatch, capsys):
    raw = ('{"feasible": true, "replacement": "Sue Pond", "edits": [\n'
           '  {"find": "in Bob Pond", "replace": "in Sue Pond"}  // rename\n'
           ']}')
    c = _rewrite_client(raw, monkeypatch)
    res = c.rewrite_context(context="chlorophyll rose in Bob Pond last year",
                            instruction="i", original="Bob Pond")
    assert res.applied and res.new_context == "chlorophyll rose in Sue Pond last year"
    assert "line comment" in capsys.readouterr().out


def test_rewrite_repairs_unescaped_latex_in_find_then_applies(monkeypatch):
    raw = ('{"feasible": true, "replacement": "0.9", "edits": '
           '[{"find": "ratio \\( r = 0.5 \\)", "replace": "ratio \\( r = 0.9 \\)"}]}')
    c = _rewrite_client(raw, monkeypatch)
    res = c.rewrite_context(context="the ratio \\( r = 0.5 \\) was noted",
                            instruction="i", original="0.5")
    assert res.applied and res.new_context == "the ratio \\( r = 0.9 \\) was noted"


def test_rewrite_fresh_miss_malformed_json_stays_out_of_cache(monkeypatch, tmp_path):
    raw = '{"feasible": true, "replacement": "b", "edits": [{"find": "a", "replace": "b"}, {"find": "c"'
    c = _rewrite_client(raw, monkeypatch, cache_path=tmp_path / "cache.json")
    with pytest.raises(RuntimeError, match=r"1/1 gpt-oss call\(s\) failed"):
        c.rewrite_context(context="a c", instruction="i", original="a")
    assert c.cache._store == {}


def test_rewrite_cache_key_carries_protocol_and_attempt(monkeypatch):
    c = _rewrite_client(json.dumps({"feasible": True, "replacement": "b"}), monkeypatch)
    old_key = pa._cache_key("rewrite", {"context": "ctx", "instruction": "i", "protocol": 2})
    c.cache._store[old_key] = json.dumps({"feasible": True, "context": "<whole page>"})
    with pytest.raises(ValueError, match="proposed no edits"):
        c.rewrite_context(context="ctx", instruction="i", original="c", attempt=0)
    k0 = pa._cache_key("rewrite", {"context": "c", "instruction": "i", "protocol": 3, "attempt": 0})
    k1 = pa._cache_key("rewrite", {"context": "c", "instruction": "i", "protocol": 3, "attempt": 1})
    assert k0 != k1


def test_rewrite_cache_key_carries_context_scope(monkeypatch):
    """A cached page-scoped response can never satisfy a full-paper request:
    the scope is in the payload, so the prompt reword cannot be silently
    absorbed by a warm cache from an earlier scope."""
    seen: dict = {}
    orig = pa._cache_key
    monkeypatch.setattr(pa, "_cache_key",
                        lambda op, payload: seen.update(payload) or orig(op, payload))
    c = _rewrite_client(json.dumps({"feasible": True, "replacement": "b",
                                    "edits": [{"find": "c", "replace": "b"}]}), monkeypatch)
    c.rewrite_context(context="a c", instruction="i", original="c")
    assert seen["context_scope"] == "full"
    full = orig("rewrite", dict(seen))
    page = orig("rewrite", {**seen, "context_scope": "page"})
    assert full != page


# ─── _extract_json_object: malformed gpt-oss responses ────────────────────────


def test_extract_json_object_dumps_raw_text_on_parse_failure(tmp_path):
    bad = '{"feasible": true, "edits": [{"find": "old", "replace": "new"}, ...]}'
    dump_dir = tmp_path / "bad_responses"
    with pytest.raises(ValueError, match="not valid JSON"):
        pa._extract_json_object(bad, op="rewrite", dump_dir=dump_dir)
    dumped = list(dump_dir.glob("bad_response_rewrite_*.txt"))
    assert len(dumped) == 1 and dumped[0].read_text() == bad


def test_extract_json_object_raises_without_dump_dir():
    with pytest.raises(ValueError, match="not valid JSON"):
        pa._extract_json_object('{"a": [1, ...]}', op="rewrite", dump_dir=None)


def test_extract_json_object_valid_json_untouched():
    obj = pa._extract_json_object('{"feasible": false, "reason": "x"}',
                                  op="rewrite", dump_dir=None)
    assert obj == {"feasible": False, "reason": "x"}


def test_extract_json_object_valid_json_does_not_dump(tmp_path):
    dump_dir = tmp_path / "bad_responses"
    pa._extract_json_object('{"feasible": true, "edits": []}', op="rewrite", dump_dir=dump_dir)
    assert not dump_dir.exists()


def test_extract_json_object_repairs_unescaped_latex_backslash():
    bad = ('{"feasible": true, "edits": [{"find": "Chlorophyll \\( a \\) rose", '
           '"replace": "Chlorophyll \\( b \\) rose"}]}')
    obj = pa._extract_json_object(bad, op="rewrite", dump_dir=None)
    assert obj["edits"][0]["find"] == "Chlorophyll \\( a \\) rose"


def test_extract_json_object_repair_leaves_valid_escapes_alone():
    bad = ('{"feasible": true, "edits": [{"find": "(\\\\( r = 0.5 \\\\))", '
           '"replace": "(\\\\( r = 0.9 \\\\))"}]}')
    obj = pa._extract_json_object(bad, op="rewrite", dump_dir=None)
    assert obj["edits"][0]["find"] == "(\\( r = 0.5 \\))"


def test_extract_json_object_repair_does_not_rescue_truncation_no_brace(tmp_path):
    bad = '{"feasible": true, "edits": [{"find": "a long verbatim span that got cut'
    dump_dir = tmp_path / "bad_responses"
    with pytest.raises(ValueError, match="not JSON"):
        pa._extract_json_object(bad, op="rewrite", dump_dir=dump_dir)
    assert len(list(dump_dir.glob("bad_response_rewrite_*.txt"))) == 1


def test_extract_json_object_repair_does_not_rescue_truncation_inner_brace(tmp_path):
    bad = '{"feasible": true, "edits": [{"find": "b"}, {"replace":'
    dump_dir = tmp_path / "bad_responses"
    with pytest.raises(ValueError, match="not valid JSON"):
        pa._extract_json_object(bad, op="rewrite", dump_dir=dump_dir)
    assert len(list(dump_dir.glob("bad_response_rewrite_*.txt"))) == 1


def test_strip_json_line_comments_removes_comments_outside_strings():
    src = '{\n  "a": 1,  // trailing note\n  "b": [2, 3]  // another\n}'
    assert json.loads(pa._strip_json_line_comments(src)) == {"a": 1, "b": [2, 3]}


def test_strip_json_line_comments_preserves_double_slash_inside_a_string():
    src = '{"context": "see https://example.org/x for detail", "n": 1}'
    assert pa._strip_json_line_comments(src) == src


def test_strip_json_line_comments_noop_on_clean_json():
    src = '{"feasible": true, "edits": [{"find": "a / b", "replace": "c / d"}]}'
    assert pa._strip_json_line_comments(src) == src


def test_extract_json_object_repairs_js_line_comments_on_edits(capsys):
    bad = (
        '{\n  "feasible": true,\n  "edits": [\n'
        '    {"find": "<td>S3</td>", "replace": "<td>Coastal Site Bravo</td>"},  // cell\n'
        '    {"find": "sites S1-S3", "replace": "sites S1-Coastal Site Bravo"} // last\n'
        '  ]\n}'
    )
    obj = pa._extract_json_object(bad, op="rewrite", dump_dir=None)
    assert obj["feasible"] is True and len(obj["edits"]) == 2
    assert "line comment" in capsys.readouterr().out


def test_extract_json_object_comment_strip_does_not_rescue_repetition_loop(tmp_path):
    bad = ('{"feasible": true, "edits": ['
           + '{"find": "WCD", "replace": "ILBB"}, ' * 40 + '{"find": "WCD", "replace": "IL')
    dump_dir = tmp_path / "bad_responses"
    with pytest.raises(ValueError, match="not valid JSON|not JSON"):
        pa._extract_json_object(bad, op="rewrite", dump_dir=dump_dir)
    assert len(list(dump_dir.glob("bad_response_rewrite_*.txt"))) == 1


def test_run_batch_keeps_unparseable_response_out_of_the_cache(tmp_path, monkeypatch):
    async def _one(self, client, messages):
        tag = messages[0]["content"]
        if tag == "loop":
            return '{"feasible": true, "edits": [{"find": "a", "replace": "b"}, {"find": "a"'
        if tag == "commented":
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
    assert set(client.cache._store) == {"k_ok", "k_commented"}
    assert json.loads(cache_path.read_text()) == client.cache._store


# ─── stub client ─────────────────────────────────────────────────────────────


def test_stub_rewrite_uses_hint_and_flags_missing():
    stub = pa.StubAugmentClient()
    res = stub.rewrite_context(context="Lake Bob has pH 7", instruction="x",
                               original="Bob", stub_replacement="Sue")
    assert res.applied and res.new_context == "Lake Sue has pH 7" and res.replacement == "Sue"
    res2 = stub.rewrite_context(context="Lake Bob has pH 7", instruction="x",
                                original="Zzz", stub_replacement="Sue")
    assert not res2.applied and "not in context" in res2.reason
    res3 = stub.rewrite_context(context="c", instruction="x", original="c",
                                stub_replacement=None)
    assert not res3.applied
    # inject case
    res4 = stub.rewrite_context(context="value 5 stated here", instruction="x",
                                original="", stub_replacement="Summer 2019")
    assert res4.applied and res4.unverified_span and "Summer 2019" in res4.new_context


# ─── fixture ─────────────────────────────────────────────────────────────────


def _rules() -> pa.DatasetAugmentRules:
    return pa.DatasetAugmentRules(
        name="toy",
        entity_name_field="name",
        entity_noun="water body",
        fabricated_names_by_type={"pond": ["Alpha Pond", "Beta Pond"],
                                  "lake": ["Gamma Lake"]},
        fabricated_names_any=["Fallback Water"],
        entity_type_token=lambda r: ("lake" if "lake" in (r.get("ecosystem") or "").lower()
                                     else "pond"),
        name_suffix_to_type={},          # toy dataset cannot verify a proposed name
        entity_preserve_clause="Keep the ecosystem type and every measured quantity identical.",
        entity_swap_clear_fields=(),
        attr_units={"tn": ["µg/L", "mg/L", "ppb"], "tp": ["µg/L", "mg/L", "ppb"]},
        attribute_pool=["tn", "tp"],
        value_pool_by_attr={"tn": ["1.0", "2.0", "3.0", "4.0", "5.0"],
                            "tp": ["10.0", "20.0", "30.0", "40.0", "50.0"]},
        event_field="date",
        event_noun="measurement date",
        event_pool=["Spring 2019", "Summer 2019", "Autumn 2019", "2020", "2021"],
        event_allow_inject=True,
        gt_cols=["document_id", "name", "ecosystem", "attribute", "value", "units",
                 "page_number", "date"],
    )


def _pond_like_rules() -> pa.DatasetAugmentRules:
    """Like ``_rules`` but with a real name→type map, so ``entity_type_ok`` bites."""
    r = _rules()
    r.name_suffix_to_type = {"pond": "pond", "lake": "lake", "water": "pond"}
    return r


def _no_event_rules() -> pa.DatasetAugmentRules:
    """Like ``_rules`` but with no judge-visible event field (supermat shape)."""
    r = _rules()
    return pa.DatasetAugmentRules(
        **{**r.__dict__, "event_field": None, "event_pool": [],
           "event_allow_inject": False, "event_noun": "n/a"}
    )


# ─── event_field = None (dataset with no judge-visible event) ─────────────────


def test_rules_reject_half_configured_event():
    """event_field None must come with an empty pool and no injection."""
    r = _rules()
    for bad in ({"event_field": None, "event_pool": ["x"]},
                {"event_field": None, "event_pool": [], "event_allow_inject": True}):
        with pytest.raises(AssertionError):
            pa.DatasetAugmentRules(**{**r.__dict__, "event_allow_inject": False, **bad})


def test_no_event_field_skips_pos_event_and_event_negative():
    rules, rng = _no_event_rules(), random.Random(0)
    src = pa._prep_base_valid(_valid_record(0, "Site 0", "pond", "tp", "10.0", "µg/L"), rules)
    assert pa.make_axis2_positive(src, "pos_event", rules, pa.StubAugmentClient(), rng) is None
    assert pa.make_typed_negative(src, "event", rules, rng) is None
    assert "event" not in pa.active_error_types(rules, [src])


def test_build_drops_pos_event_when_no_event_field(monkeypatch):
    """The orchestrator strips pos_event from the axis set for a no-event dataset."""
    rules = _no_event_rules()
    seen: list[tuple[str, ...]] = []

    class _Stop(Exception):
        pass

    def _spy(gt_valids, rules_, client, rng, *, floor, axes, **kw):
        seen.append(axes)
        raise _Stop            # only the axis set matters here

    monkeypatch.setattr(pa, "fill_positive_target", _spy)
    flags = pa.AugmentFlags(pos_axes=("pos_entity", "pos_value", "pos_event"),
                            valid_floor=0, diag_valid_floor=0)
    with pytest.raises(_Stop):
        pa.build_augmented_files(
            xv_train=[_valid_record(0, "Site 0", "pond", "tp", "10.0", "µg/L")],
            xv_test=[_valid_record(1, "Site 1", "lake", "tn", "11.0", "µg/L")],
            rng=random.Random(0), client=pa.StubAugmentClient(), rules=rules,
            flags=flags, quiet=True)
    assert seen == [("pos_entity", "pos_value")]


def _valid_record(i: int, name, eco: str, attr: str, val: str, units: str) -> dict:
    return {
        "document_id": f"D{i % 3}", "name": name, "ecosystem": eco,
        "attribute": attr, "value": val, "units": units, "page_number": 1, "date": None,
        "gt_row_index": i, "donor_gt_row_index": None,
        "_orig_idx": i, "_paper_code": f"D{i % 3}", "_page_numbers": [1],
        pa._CTX_ORIG_KEY: f'The site "{name}" reported {attr} of "{val}" {units} on page one.',
    }


def _fixture_valids(n: int) -> list[dict]:
    ecos = ["shallow pond", "karst lake"]
    return [_valid_record(i, f"Site {i}", ecos[i % 2], "tn" if i % 2 else "tp",
                          f"{10 + i}.0", "µg/L")
            for i in range(n)]


# ─── entity_type_ok ──────────────────────────────────────────────────────────


def test_entity_type_ok_only_bites_with_a_name_map():
    toy = _rules()
    src = _valid_record(0, "Site 0", "shallow pond", "tp", "10.0", "µg/L")
    assert toy.entity_type_ok(src, "Anything At All") is True     # no map
    pond = _pond_like_rules()
    assert pond.entity_type_ok(src, "Silver Pond") is True        # matches type
    assert pond.entity_type_ok(src, "Gamma Lake") is False        # wrong type
    assert pond.entity_type_ok(src, "Nowhere Bluff") is False     # unrecognised suffix


# ─── axis-2 positives ────────────────────────────────────────────────────────


def test_axis2_pos_entity_edits_context_and_measurement():
    rules, rng = _rules(), random.Random(0)
    src = pa._prep_base_valid(_valid_record(0, "Site 0", "shallow pond", "tp", "10.0", "µg/L"),
                              rules)
    row = pa.make_axis2_positive(src, "pos_entity", rules, pa.StubAugmentClient(), rng)
    assert row is not None
    assert row["label"] == "valid" and row["augment_axis"] == "pos_entity"
    assert row["name"] in {"Alpha Pond", "Beta Pond"}       # same-type stub hint
    assert row["name"] in row[pa._CTX_EDIT_KEY] and "Site 0" not in row[pa._CTX_EDIT_KEY]
    assert row["source_group_id"] == 0
    assert "measurement_id" not in row          # assigned per-file, after assembly


def test_axis2_pos_value_changes_the_number():
    rules, rng = _rules(), random.Random(0)
    src = pa._prep_base_valid(_valid_record(1, "Site 1", "karst lake", "tn", "11.0", "µg/L"),
                              rules)
    row = pa.make_axis2_positive(src, "pos_value", rules, pa.StubAugmentClient(), rng)
    assert row is not None and row["augment_axis"] == "pos_value"
    assert row["value"] != "11.0" and row["value"] in row[pa._CTX_EDIT_KEY]


def test_axis2_pos_event_injects_when_gt_event_is_null():
    rules, rng = _rules(), random.Random(0)
    src = pa._prep_base_valid(_valid_record(2, "Site 2", "shallow pond", "tp", "12.0", "µg/L"),
                              rules)
    row = pa.make_axis2_positive(src, "pos_event", rules, pa.StubAugmentClient(), rng)
    assert row is not None and row["augment_axis"] == "pos_event"
    assert row["date"] in rules.event_pool
    assert row["_unverified_span"] is True          # an injected event is not page-grounded


def test_axis2_pos_event_skips_inject_when_not_allowed():
    rules, rng = _rules(), random.Random(0)
    rules.event_allow_inject = False
    src = pa._prep_base_valid(_valid_record(2, "Site 2", "shallow pond", "tp", "12.0", "µg/L"),
                              rules)
    assert pa.make_axis2_positive(src, "pos_event", rules, pa.StubAugmentClient(), rng) is None


def test_axis2_pos_entity_type_mismatch_is_skipped(monkeypatch):
    rules, rng = _pond_like_rules(), random.Random(0)
    src = pa._prep_base_valid(_valid_record(0, "Site 0", "shallow pond", "tp", "10.0", "µg/L"),
                              rules)
    raw = json.dumps({"feasible": True, "replacement": "Gamma Lake", "edits": [
        {"find": 'The site "Site 0" reported', "replace": 'The site "Gamma Lake" reported'}]})

    async def _one(self, client, messages):
        return raw
    monkeypatch.setattr(pa.GptOssClient, "_one", _one)
    client = pa.GptOssClient(api_base="x", cache=pa.AugmentCache(None))
    assert pa.make_axis2_positive(src, "pos_entity", rules, client, rng) is None


# ─── typed hard negatives ────────────────────────────────────────────────────


def test_typed_negative_entity_from_pool():
    rules, rng = _rules(), random.Random(0)
    src = pa._prep_base_valid(_valid_record(0, "Site 0", "shallow pond", "tp", "10.0", "µg/L"),
                              rules)
    neg = pa.make_typed_negative(src, "entity", rules, rng)
    assert neg["label"] == "invalid" and neg["modification_type"] == "bad_entity"
    assert neg["name"] in {"Alpha Pond", "Beta Pond"} and neg["name"] != "Site 0"


def test_typed_negative_attribute_needs_two_attributes():
    rules, rng = _rules(), random.Random(0)
    src = pa._prep_base_valid(_valid_record(0, "S", "pond", "tp", "10.0", "µg/L"), rules)
    assert pa.make_typed_negative(src, "attribute", rules, rng)["attribute"] == "tn"
    rules.attribute_pool = ["tp"]
    assert pa.make_typed_negative(src, "attribute", rules, rng) is None


def test_typed_negative_value_avoids_context_collision():
    rules = _rules()
    # context states "10.0"; the tp pool is 10/20/30/40/50 -> never draw 10.0
    src = pa._prep_base_valid(_valid_record(0, "S", "pond", "tp", "10.0", "µg/L"), rules)
    for s in range(30):
        neg = pa.make_typed_negative(src, "value", rules, random.Random(s))
        assert neg["value"] in {"20.0", "30.0", "40.0", "50.0"}


def test_typed_negative_units_swaps():
    rules, rng = _rules(), random.Random(0)
    src = pa._prep_base_valid(_valid_record(0, "S", "pond", "tp", "10.0", "mg/L"), rules)
    neg = pa.make_typed_negative(src, "units", rules, rng)
    assert neg["units"] in {"µg/L", "ppb"}


def test_typed_negative_event_skips_when_source_event_null():
    rules, rng = _rules(), random.Random(0)
    src = pa._prep_base_valid(_valid_record(0, "S", "pond", "tp", "10.0", "µg/L"), rules)
    assert src.get("date") is None
    assert pa.make_typed_negative(src, "event", rules, rng) is None
    src["date"] = "Spring 2019"
    neg = pa.make_typed_negative(src, "event", rules, rng)
    assert neg["date"] != "Spring 2019" and neg["modification_type"] == "bad_event"


def test_typed_negative_entity_skips_on_null_entity_field():
    rules, rng = _rules(), random.Random(0)
    src = pa._prep_base_valid(_valid_record(0, None, "pond", "tp", "10.0", "µg/L"), rules)
    assert pa.make_typed_negative(src, "entity", rules, rng) is None
    # a stated numeric claim can still be corrupted
    assert pa.make_typed_negative(src, "value", rules, rng) is not None


def test_typed_negative_on_synthetic_positive_carries_edited_context():
    rules, rng = _rules(), random.Random(0)
    src = pa._prep_base_valid(_valid_record(0, "Site 0", "shallow pond", "tp", "10.0", "µg/L"),
                              rules)
    pos = pa.make_axis2_positive(src, "pos_entity", rules, pa.StubAugmentClient(), rng)
    neg = pa.make_typed_negative(pos, "value", rules, rng)
    assert neg[pa._CTX_EDIT_KEY] == pos[pa._CTX_EDIT_KEY]
    assert neg["augment_axis"] == "pos_entity"


def test_draw_alt_rejects_context_present_entries():
    rng = random.Random(0)
    assert pa._draw_alt(["a", "b", "c"], "a", rng, "text has b and c inside") is None
    assert pa._draw_alt(["a", "b", "c"], "a", rng, "text has c only") == "b"


# ─── quotas + fillers ────────────────────────────────────────────────────────


def test_even_quota_splits_the_remainder():
    assert pa.even_quota(1000, list("abcde")) == {k: 200 for k in "abcde"}
    q = pa.even_quota(101, list("abc"))
    assert sum(q.values()) == 101 and sorted(q.values()) == [33, 34, 34]


def test_active_error_types_gates_attribute_and_event():
    rules = _rules()
    valids = [pa._prep_base_valid(v, rules) for v in _fixture_valids(4)]
    assert pa.active_error_types(rules, valids) == ["entity", "attribute", "value", "units"]
    valids[0]["date"] = "Spring 2019"
    assert pa.active_error_types(rules, valids) == \
        ["entity", "attribute", "value", "units", "event"]
    rules.attribute_pool = ["tp"]
    assert pa.active_error_types(rules, valids) == ["entity", "value", "units", "event"]


def test_fill_negative_quota_hits_exact_counts_and_dedups():
    rules, rng = _rules(), random.Random(0)
    valids = [pa._prep_base_valid(v, rules) for v in _fixture_valids(40)]
    for v in valids:
        v["date"] = "Spring 2019"
    quota = {"entity": 10, "value": 10, "units": 10, "event": 10}
    negs = pa.fill_negative_quota(valids, rules, rng, quota_by_type=quota, seen_sigs=set())
    got: dict[str, int] = {}
    for n in negs:
        got[n["modification_type"]] = got.get(n["modification_type"], 0) + 1
    assert got == {"bad_entity": 10, "bad_value": 10, "bad_units": 10, "bad_event": 10}
    assert all(n["label"] == "invalid" for n in negs)
    sigs = [pa._row_signature(n, rules.gt_cols) for n in negs]
    assert len(sigs) == len(set(sigs))                 # no repeat invalid entries


def test_fill_negative_quota_fails_loud_on_missing_pool():
    rules, rng = _rules(), random.Random(0)
    rules.event_pool = []
    valids = [pa._prep_base_valid(v, rules) for v in _fixture_valids(20)]
    for v in valids:
        v["date"] = "Spring 2019"
    with pytest.raises(RuntimeError, match="event.*pool is missing"):
        pa.fill_negative_quota(valids, rules, rng, quota_by_type={"event": 5}, seen_sigs=set())


def test_fill_negative_quota_fails_loud_when_pool_too_shallow_for_quota():
    rules, rng = _rules(), random.Random(0)
    rules.attr_units = {"tp": ["µg/L", "mg/L"]}         # one alternative per source
    v = pa._prep_base_valid(_valid_record(0, "S", "pond", "tp", "10.0", "µg/L"), rules)
    with pytest.raises(RuntimeError, match="duplicate draws"):
        pa.fill_negative_quota([v], rules, rng, quota_by_type={"units": 5}, seen_sigs=set())


def test_fill_positive_target_round0_keeps_everything():
    rules, rng = _rules(), random.Random(0)
    gt = [pa._prep_base_valid(v, rules) for v in _fixture_valids(12)]
    seen = {pa._row_signature(r, rules.gt_cols) for r in gt}
    out = pa.fill_positive_target(gt, rules, pa.StubAugmentClient(), rng, floor=5,
                                  axes=pa.POS_AXES, prompt_budget_multiple=1, seen_sigs=seen)
    # round 0 = 12 sources × 3 axes, all accepted -> well above the floor of 5
    assert len(out) > 5 and all(r["label"] == "valid" for r in out)
    sigs = [pa._row_signature(r, rules.gt_cols) for r in out]
    assert len(sigs) == len(set(sigs))                 # no repeat valid entries


def test_fill_positive_target_fails_loud_when_floor_unreachable():
    # the stub is deterministic, so a (source, axis) yields one distinct row ever;
    # 6 sources × 1 axis caps distinct positives at 6, well under the floor.
    rules, rng = _rules(), random.Random(0)
    gt = [pa._prep_base_valid(v, rules) for v in _fixture_valids(6)]
    seen = {pa._row_signature(r, rules.gt_cols) for r in gt}
    with pytest.raises(RuntimeError, match="positive floor not met"):
        pa.fill_positive_target(gt, rules, pa.StubAugmentClient(), rng, floor=100,
                                axes=("pos_entity",), prompt_budget_multiple=5, seen_sigs=seen)


# ─── context-override inlining + full pipeline determinism ───────────────────


def test_inline_context_overrides_stamps_every_row():
    rows = [
        {"measurement_id": 0, pa._CTX_ORIG_KEY: "a", pa._CTX_EDIT_KEY: "A"},  # genuine edit
        {"measurement_id": 1, pa._CTX_ORIG_KEY: "b"},                         # unedited
        {"measurement_id": 2, pa._CTX_ORIG_KEY: "c", pa._CTX_EDIT_KEY: "c"},  # no-op edit
    ]
    assert pa.inline_context_overrides(rows) == 1          # only row 0 is a genuine edit
    assert rows[0]["context_override"] == "A"              # edited paper
    assert rows[1]["context_override"] == "b"              # verbatim source paper
    assert rows[2]["context_override"] == "c"              # no-op edit falls back to source


def test_inline_context_overrides_requires_a_source_context():
    with pytest.raises(AssertionError, match="no source context"):
        pa.inline_context_overrides([{"measurement_id": 9}])


def _run_pipeline(seed: int):
    rules = _rules()
    flags = pa.AugmentFlags(pos_axes=pa.POS_AXES, valid_floor=26,
                            diag_valid_floor=8, prompt_budget_multiple=1)
    valids = _fixture_valids(30)
    tr, te = valids[:20], valids[20:]
    return pa.build_augmented_files(
        xv_train=[dict(v) for v in tr], xv_test=[dict(v) for v in te],
        rng=random.Random(seed), client=pa.StubAugmentClient(), rules=rules, flags=flags,
    )


def test_pipeline_is_seed_deterministic():
    a, b = _run_pipeline(42), _run_pipeline(42)
    for key in ("train", "primary_test", "diagnostic_test"):
        assert a[key][0] == b[key][0], f"{key} rows differ across two seed-42 runs"
        assert a[key][1] == b[key][1], f"{key} raw rows differ across two seed-42 runs"


def test_pipeline_files_are_wellformed():
    out = _run_pipeline(42)
    for key in ("train", "primary_test", "diagnostic_test"):
        rows, _ = out[key]
        pa.assert_wellformed(key, rows)                # 50/50, contiguous ids
        assert all(r.get("source_group_id") is not None for r in rows)
        sigs = [pa._row_signature(r, _rules().gt_cols) for r in rows]
        assert len(sigs) == len(set(sigs)), f"{key} has repeat entries"
    train_rows, _ = out["train"]
    assert len(train_rows) >= 52                       # floor 26 × 2, usually more
    # Every row carries the full-paper context_override, valid and invalid,
    # edited and unedited, in every file — including the zero-gpt-oss primary test.
    for key in ("train", "primary_test", "diagnostic_test"):
        rows, raw = out[key]
        assert all(r["context_override"] for r in rows), key
    d_rows, d_raw = out["diagnostic_test"]
    assert any(r.get(pa._CTX_EDIT_KEY) for r in d_raw)          # some synthetic positives (edited paper)
    assert any(not r.get(pa._CTX_EDIT_KEY) for r in d_raw)      # some carried GT valids (verbatim paper)


def test_assert_wellformed_rejects_a_row_without_a_context_override():
    rows, _ = _run_pipeline(42)["train"]
    pa.assert_wellformed("train", rows)                         # baseline: passes
    rows[3] = {k: v for k, v in rows[3].items() if k != "context_override"}
    with pytest.raises(AssertionError, match="full-paper context"):
        pa.assert_wellformed("train", rows)


def test_run_and_write_inlines_overrides_no_sidecar_file(tmp_path):
    import judge_common
    rules = _rules()
    flags = pa.AugmentFlags(pos_axes=pa.POS_AXES, valid_floor=26,
                            diag_valid_floor=8, prompt_budget_multiple=1)
    valids = _fixture_valids(30)
    ocr_dir = tmp_path / "ocr"
    ocr_dir.mkdir()
    by_code: dict[str, list[dict]] = {}
    for v in valids:
        by_code.setdefault(v["_paper_code"], []).append(v)
    # Two pages per paper: the row's page_number is [1], page 2 carries a marker
    # that appears nowhere on page 1 — so a page-sliced context would drop it.
    marker = "ZZ_PAGE_TWO_ONLY_MARKER"
    for code, vs in by_code.items():
        body = " ".join(f'The site "{v["name"]}" reported {v["attribute"]} of '
                        f'"{v["value"]}" {v["units"]}.' for v in vs)
        (ocr_dir / f"{code}.txt").write_text(
            f'<page number="1">{body}</page>\n<page number="2">{marker}</page>')

    written = pa.run_and_write(
        base_dir=tmp_path, out_suffix="_v2", ocr_dir=ocr_dir,
        xv_train=[dict(v) for v in valids[:20]], xv_test=[dict(v) for v in valids[20:]],
        rules=rules, flags=flags, client=pa.StubAugmentClient(), rng=random.Random(42),
    )
    train_p, primary_p, diag_p = (written["train"], written["primary_test"],
                                  written["diagnostic_test"])
    assert not list(tmp_path.glob("probe_context_overrides*"))
    assert (tmp_path / (train_p.name + ".diff.txt")).exists()
    assert (tmp_path / (diag_p.name + ".diff.txt")).exists()
    assert not (tmp_path / (primary_p.name + ".diff.txt")).exists()

    train_data = json.loads(train_p.read_text())
    diag_data = json.loads(diag_p.read_text())
    primary_data = json.loads(primary_p.read_text())
    documents = {v["_paper_code"]: (ocr_dir / f"{v['_paper_code']}.txt").read_text()
                for v in valids}

    # Every row in every file carries the full paper as context_override —
    # unedited rows verbatim, edited rows a variant that still contains the
    # untouched page-2 marker. No row is page-sliced.
    for data in (train_data, diag_data, primary_data):
        assert data
        for r in data:
            override = r["context_override"]
            assert marker in override, r["measurement_id"]
            if r.get("modification_type") is None and r.get("augment_axis") is None:
                assert override == documents[str(r["document_id"])]   # carried GT valid, verbatim

    # The real invariant: an edit changes the paper without truncating it to a
    # page. Some train rows differ from their source document, and every such row
    # still carries the full paper — the page-2 marker included.
    edited = [r for r in train_data
              if r["context_override"] != documents[str(r["document_id"])]]
    assert edited and all(marker in r["context_override"] for r in edited)

    cfg = _dcfg()
    for data in (train_data, diag_data, primary_data):
        entries = judge_common.prepare_chat_entries(data, documents, cfg)
        for entry in entries:
            row = data[int(entry["custom_id"])]
            assert entry["context_text"] == row["context_override"]


# ─── GptOssClient prewarm / record-pass plumbing ─────────────────────────────


def _canned_run_batch():
    """An async `_run_batch` stand-in (no server). Protocol-3 responses: anchor
    on the first double-quoted token that occurs verbatim in the paper text; a
    'does not state' instruction is the pos_event inject case."""

    async def _run_batch(self, jobs):
        out: dict[str, str] = {}
        for key, messages in jobs:
            instr, paper = messages[-1]["content"].split("## PAPER TEXT\n", 1)
            anchor = next((q for q in re.findall(r'"([^"]+)"', instr) if q in paper), None)
            if anchor is None:
                out[key] = json.dumps({"feasible": False, "reason": "canned: no anchor"})
            elif "does not state" in instr:
                repl = "Summer 2019"
                out[key] = json.dumps({"feasible": True, "replacement": repl,
                    "edits": [{"find": anchor, "replace": f"{anchor} ({repl})"}]})
            else:
                repl = (anchor[::-1] or "Z") + "x"
                out[key] = json.dumps({"feasible": True, "replacement": repl,
                    "edits": [{"find": anchor, "replace": repl}]})
        for k, v in out.items():
            self.cache.put(k, v)
        return out

    return _run_batch


def _augment_fixture(axes=pa.POS_AXES):
    rules = _rules()
    flags = pa.AugmentFlags(pos_axes=axes, valid_floor=26, diag_valid_floor=8,
                            prompt_budget_multiple=1)
    valids = _fixture_valids(30)
    return ([dict(v) for v in valids[:20]], [dict(v) for v in valids[20:]], rules, flags)


def _record_pass(client, tr, te, rules, flags, seed=42):
    rng = random.Random(seed)
    state = rng.getstate()
    client.record = True
    pa.build_augmented_files(xv_train=tr, xv_test=te, rng=rng, client=client,
                             rules=rules, flags=flags, quiet=True)
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
    assert all(re.fullmatch(r"[0-9a-f]{64}", k) for k, _ in jobs)
    assert client.cache._store == {} and client.cache._misses == 0


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
    out = asyncio.run(client._one(_FakeClient(_FakeChoice("hello"), captured),
                                  [{"role": "user", "content": "hi"}]))
    assert out == "hello"
    assert captured["extra_body"] == {"chat_template_kwargs": {"reasoning_effort": "low"}}


def test_one_empty_content_error_carries_finish_reason_and_reasoning_length():
    client = pa.GptOssClient(api_base="x", cache=pa.AugmentCache(None))
    fake = _FakeClient(_FakeChoice(None, finish_reason="length", reasoning_content="x" * 500), {})
    with pytest.raises(ValueError, match=r"finish_reason='length'.*reasoning_content_len=500"):
        asyncio.run(client._one(fake, [{"role": "user", "content": "hi"}]))


def test_one_raises_on_length_finish_with_nonempty_content():
    client = pa.GptOssClient(api_base="x", cache=pa.AugmentCache(None))
    fake = _FakeClient(_FakeChoice('{"feasible": tr', finish_reason="length"), {})
    with pytest.raises(ValueError, match=r"truncated at max_tokens.*finish_reason='length'"):
        asyncio.run(client._one(fake, [{"role": "user", "content": "hi"}]))


def test_run_batch_partial_failure_caches_successes_before_raising(tmp_path, monkeypatch):
    async def _one(self, client, messages):
        if messages[0]["content"] == "fail":
            raise ValueError("simulated gpt-oss failure")
        return json.dumps({"feasible": True, "replacement": "x", "edits": []})
    monkeypatch.setattr(pa.GptOssClient, "_one", _one)
    cache_path = tmp_path / "cache.json"
    client = pa.GptOssClient(api_base="x", cache=pa.AugmentCache(cache_path))
    jobs = [(f"k{i}", [{"role": "user", "content": "fail" if i == 1 else "ok"}])
            for i in range(4)]
    with pytest.raises(RuntimeError, match=r"1/4 gpt-oss call\(s\) failed"):
        asyncio.run(client._run_batch(jobs))
    assert set(client.cache._store) == {"k0", "k2", "k3"}
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
    assert p1.read_bytes() == p2.read_bytes()


def test_prewarm_cache_survives_a_crash_in_the_post_prewarm_assembly(tmp_path, monkeypatch):
    monkeypatch.setattr(pa.GptOssClient, "_run_batch", _canned_run_batch())
    _, _, rules, flags = _augment_fixture()
    valids = _fixture_valids(30)
    ocr_dir = tmp_path / "ocr"
    ocr_dir.mkdir()
    by_code: dict[str, list[dict]] = {}
    for v in valids:
        by_code.setdefault(v["_paper_code"], []).append(v)
    for code, vs in by_code.items():
        body = " ".join(f'The site "{v["name"]}" reported {v["attribute"]} of '
                        f'"{v["value"]}" {v["units"]}.' for v in vs)
        (ocr_dir / f"{code}.txt").write_text(f'<page number="1">{body}</page>')

    cache_path = tmp_path / "cache.json"
    client = pa.GptOssClient(api_base="x", cache=pa.AugmentCache(cache_path))
    real_build = pa.build_augmented_files

    def _boom_after_prewarm(*, client, **kwargs):
        if not client.record:
            raise RuntimeError("simulated crash in assembly, after prewarm")
        return real_build(client=client, **kwargs)
    monkeypatch.setattr(pa, "build_augmented_files", _boom_after_prewarm)

    with pytest.raises(RuntimeError, match="simulated crash"):
        pa.run_and_write(
            base_dir=tmp_path, out_suffix="_v2", ocr_dir=ocr_dir,
            xv_train=[dict(v) for v in valids[:20]], xv_test=[dict(v) for v in valids[20:]],
            rules=rules, flags=flags, client=client, rng=random.Random(42),
        )
    assert cache_path.exists()
    saved = json.loads(cache_path.read_text())
    assert saved == client.cache._store and len(saved) > 0


@pytest.mark.parametrize("resumed", [False, True])
def test_record_then_real_pass_makes_zero_model_calls(monkeypatch, resumed):
    monkeypatch.setattr(pa.GptOssClient, "_run_batch", _canned_run_batch())
    tr, te, rules, flags = _augment_fixture()

    probe = pa.GptOssClient(api_base="x", cache=pa.AugmentCache(None))
    discovered, _ = _record_pass(probe, tr, te, rules, flags)
    assert discovered

    client = pa.GptOssClient(api_base="x", cache=pa.AugmentCache(None))
    if resumed:
        for k, _ in discovered[::2]:
            client.cache._store[k] = json.dumps(
                {"feasible": False, "reason": "stale (resumed-run cache)"})

    jobs, rng = _record_pass(client, tr, te, rules, flags)
    client.prewarm(jobs)
    misses_after_prewarm = client.cache._misses
    client.strict_cache = True

    out = pa.build_augmented_files(xv_train=tr, xv_test=te, rng=rng, client=client,
                                   rules=rules, flags=flags)
    assert client.cache._misses == misses_after_prewarm      # strict_cache never fell through
    for key in ("train", "primary_test", "diagnostic_test"):
        pa.assert_wellformed(key, out[key][0])


# ─── judge_common.prepare_chat_entries context-override plumbing ──────────────


def _dcfg():
    from run_extraction import load_dataset_config
    return load_dataset_config("pond")


def test_prepare_chat_entries_no_override_field_gets_full_document():
    import judge_common
    cfg = _dcfg()
    data = [{"document_id": "X", "attribute": "tp", "value": "5", "units": "µg/L",
             "measurement_id": 0, "name": "L", "page_number": [1]}]
    docs = {"X": '<page number="0">intro</page>\n\n'
                 '<page number="1">total phosphorus 5 µg/L</page>'}
    entries = judge_common.prepare_chat_entries(data, docs, cfg)
    # The whole paper is the context — page_number is not consulted.
    assert entries[0]["context_text"] == docs["X"]


def test_prepare_chat_entries_null_override_is_inert():
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
    assert entries[0]["context_text"] == "REWRITTEN CONTEXT for the probe"
    assert "REWRITTEN CONTEXT" in entries[0]["user"]


def test_prepare_chat_entries_override_is_per_row():
    import judge_common
    cfg = _dcfg()
    data = [
        {"document_id": "X", "attribute": "tp", "value": "5", "units": "µg/L",
         "measurement_id": 0, "name": "L", "page_number": [1], "context_override": "EDITED"},
        {"document_id": "X", "attribute": "tp", "value": "6", "units": "µg/L",
         "measurement_id": 1, "name": "M", "page_number": [1]},
    ]
    docs = {"X": '<page number="0">intro</page>\n\n'
                 '<page number="1">total phosphorus 5 µg/L</page>'}
    entries = judge_common.prepare_chat_entries(data, docs, cfg)
    by_mid = {int(e["custom_id"]): e for e in entries}
    assert by_mid[0]["context_text"] == "EDITED"          # override row
    assert by_mid[1]["context_text"] == docs["X"]         # non-override row → full paper


# ─── --synthetic-name path routing (experiments/paths.py) ─────────────────────


def test_synthetic_probe_named_routes_to_dedicated_tree():
    import paths
    p = paths.synthetic_probe_named("pond", "v2_diag", "mistral-7b", "2026_09_10")
    assert p.parts[-3:] == ("synthetic_probe_v2_diag", "mistral-7b", "2026_09_10")


def test_synthetic_probe_split_vs_name_precedence():
    import paths
    assert paths.synthetic_probe("pond", "m").parts[-3] == "synthetic_probe"
    assert paths.synthetic_probe_test("pond", "m").parts[-3] == "synthetic_probe_test"
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
    with pytest.raises(FileNotFoundError):
        paths.find_synthetic_responses("pond", "mistral-7b", "2026_09_10")
