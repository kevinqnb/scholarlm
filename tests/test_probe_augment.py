"""Rung-1 (model-free) unit tests for ``scholarlm.utils.probe_augment`` and the
``judge_common.prepare_chat_entries`` context-override plumbing.

Build note: ``notes/scholarlm/builds/2026-09-03-probe-synthetic-augmentation-01.md``.
Everything here runs against a hand-built fixture with the ``StubAugmentClient``
(no LLM), so the expected output is inspectable.
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

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


# ─── side-car assembly + full pipeline determinism ───────────────────────────


def test_build_context_overrides_only_edited_rows():
    rows = [
        {"measurement_id": 0, pa._CTX_ORIG_KEY: "a", pa._CTX_EDIT_KEY: "A"},
        {"measurement_id": 1, pa._CTX_ORIG_KEY: "b"},                       # unedited
        {"measurement_id": 2, pa._CTX_ORIG_KEY: "c", pa._CTX_EDIT_KEY: "c"},  # noop edit
    ]
    sc = pa.build_context_overrides(rows)
    assert sc == {"0": "A"}


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
        assert a[key][1] == b[key][1], f"{key} side-car differs across two seed-42 runs"


def test_pipeline_files_are_wellformed(tmp_path):
    out = _run_pipeline(42, tmp_path)
    for key in ("train", "primary_test", "diagnostic_test"):
        rows, side_car, _ = out[key]
        pa.assert_wellformed(key, rows, side_car)
        assert all("source_group_id" in r for r in rows)
    # diagnostic file: every row sits on an edited context
    d_rows, d_sc, _ = out["diagnostic_test"]
    assert set(d_sc) == {str(r["measurement_id"]) for r in d_rows}


def test_run_and_write_stamps_sidecars_and_load_roundtrips(tmp_path):
    """run_and_write → stamped side-car files → judge_common.load_context_overrides.

    The stamp is what closes audit List-1 #5: a side-car whose measurement_id
    range nests inside another file's must not load against the wrong file.
    """
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

    # side-car files are stamped with the data file they belong to
    for p, sc_name in [(train_p, "probe_context_overrides_v2.json"),
                       (diag_p, "probe_context_overrides_test_v2_diag.json")]:
        doc = json.loads((tmp_path / sc_name).read_text())
        assert set(doc) == {"probe_file", "overrides"}
        assert doc["probe_file"] == p.name
        data_mids = {str(r["measurement_id"]) for r in json.loads(p.read_text())}
        assert set(doc["overrides"]) <= data_mids            # resolution (c)

    # sibling auto-load: train + diag resolve, primary has none
    assert isinstance(judge_common.load_context_overrides(train_p), dict)
    assert isinstance(judge_common.load_context_overrides(diag_p), dict)
    assert judge_common.load_context_overrides(primary_p) is None

    # the audit List-1 #5 case: diag side-car explicitly loaded against the
    # primary file — nested id ranges, previously silent — now a hard error
    diag_sc = judge_common.sidecar_path_for(diag_p)
    with pytest.raises(ValueError, match="wrong side-car"):
        judge_common.load_context_overrides(primary_p, explicit_path=diag_sc)


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
    """An async `_run_batch` stand-in: stub-style rewrites, canned event-fill."""

    async def _run_batch(self, jobs):
        out: dict[str, str] = {}
        for key, messages in jobs:
            user = messages[-1]["content"]
            if "## MEASUREMENT\n" in user:                       # event_fill
                out[key] = json.dumps({} if event_date is None else {"date": event_date})
                continue
            instr, page = user.split("## PAGE TEXT\n", 1)        # rewrite
            quoted = re.findall(r'"([^"]+)"', instr)
            olds = [t for t in quoted if t in page]
            news = [t for t in quoted if t not in page]
            ctx = page
            for o in olds:
                ctx = ctx.replace(o, news[0] if news else o + "_X")
            out[key] = json.dumps({"feasible": True, "context": ctx, "replacements": []})
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
        # before the client call and do not depend on its return.
        for k in rewrite_keys[::2]:
            client.cache._store[k] = json.dumps(
                {"feasible": True, "context": "stale", "replacements": []})

    jobs, rng = _record_pass(client, tr, te, rules, flags)
    client.prewarm(jobs)
    misses_after_prewarm = client.cache._misses
    client.strict_cache = True

    out = pa.build_augmented_files(xv_train=tr, xv_test=te, rng=rng,
                                   client=client, rules=rules, flags=flags)

    # strict_cache did not raise, and not one new model call happened
    assert client.cache._misses == misses_after_prewarm
    for key in ("train", "primary_test", "diagnostic_test"):
        pa.assert_wellformed(key, out[key][0], out[key][1])


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


def test_prepare_chat_entries_none_override_is_inert():
    import judge_common
    cfg = _dcfg()
    data = [{"document_id": "X", "attribute": "tp", "value": "5", "units": "µg/L",
             "measurement_id": 0, "name": "L", "page_number": [1]}]
    docs = {"X": '<page number="1">total phosphorus 5 µg/L</page>'}
    a = judge_common.prepare_chat_entries(data, docs, cfg)
    b = judge_common.prepare_chat_entries(data, docs, cfg, context_overrides=None)
    assert a == b


def test_prepare_chat_entries_applies_override():
    import judge_common
    cfg = _dcfg()
    data = [{"document_id": "X", "attribute": "tp", "value": "5", "units": "µg/L",
             "measurement_id": 7, "name": "L", "page_number": [1]}]
    docs = {"X": '<page number="1">total phosphorus 5 µg/L</page>'}
    entries = judge_common.prepare_chat_entries(
        data, docs, cfg, context_overrides={"7": "REWRITTEN CONTEXT for the probe"}
    )
    assert entries[0]["page_text"] == "REWRITTEN CONTEXT for the probe"
    assert "REWRITTEN CONTEXT" in entries[0]["user"]


def test_prepare_chat_entries_rejects_wrong_sidecar():
    import judge_common
    cfg = _dcfg()
    data = [{"document_id": "X", "attribute": "tp", "value": "5", "units": "µg/L",
             "measurement_id": 7, "name": "L", "page_number": [1]}]
    docs = {"X": '<page number="1">x</page>'}
    with pytest.raises(ValueError):
        judge_common.prepare_chat_entries(data, docs, cfg,
                                          context_overrides={"999": "nope"})


def test_prepare_chat_entries_rejects_partial_sidecar_mismatch():
    import judge_common
    cfg = _dcfg()
    data = [{"document_id": "X", "attribute": "tp", "value": "5", "units": "µg/L",
             "measurement_id": 7, "name": "L", "page_number": [1]}]
    docs = {"X": '<page number="1">x</page>'}
    with pytest.raises(ValueError):
        judge_common.prepare_chat_entries(
            data, docs, cfg, context_overrides={"7": "ok", "8": "orphan"})


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


# ─── context-override side-car loading (judge_common) ─────────────────────────


def test_sidecar_path_for_derives_sibling_name():
    import judge_common
    for src, want in [
        ("data/pond/probe_dataset.json", "probe_context_overrides.json"),
        ("data/pond/probe_dataset_v2.json", "probe_context_overrides_v2.json"),
        ("d/probe_dataset_test_v2_diag.json", "probe_context_overrides_test_v2_diag.json"),
    ]:
        assert judge_common.sidecar_path_for(Path(src)).name == want


def test_sidecar_path_for_rejects_unrelated_name():
    import judge_common
    with pytest.raises(ValueError):
        judge_common.sidecar_path_for(Path("data/pond/something_else.json"))


def test_load_context_overrides_missing_sibling_returns_none(tmp_path):
    import judge_common
    probe = tmp_path / "probe_dataset_v2.json"
    probe.write_text("[]")
    assert judge_common.load_context_overrides(probe) is None


def test_load_context_overrides_explicit_missing_raises(tmp_path):
    import judge_common
    with pytest.raises(FileNotFoundError):
        judge_common.load_context_overrides(tmp_path / "probe_dataset.json",
                                            explicit_path=tmp_path / "no_such.json")


def test_load_context_overrides_reads_sibling_and_validates_shape(tmp_path):
    import json
    import judge_common
    probe = tmp_path / "probe_dataset_v2.json"
    probe.write_text("[]")
    sc = tmp_path / "probe_context_overrides_v2.json"
    sc.write_text(json.dumps({"probe_file": "probe_dataset_v2.json",
                              "overrides": {"3": "ctx"}}))
    assert judge_common.load_context_overrides(probe) == {"3": "ctx"}
    # non-{str: str} overrides payload is a hard error
    sc.write_text(json.dumps({"probe_file": "probe_dataset_v2.json",
                              "overrides": {"3": 5}}))
    with pytest.raises(ValueError):
        judge_common.load_context_overrides(probe)
    # a bare flat dict (the pre-stamp format) is now rejected
    sc.write_text(json.dumps({"3": "ctx"}))
    with pytest.raises(ValueError):
        judge_common.load_context_overrides(probe)


def test_load_context_overrides_rejects_wrong_probe_file_stamp(tmp_path):
    """A side-car whose measurement_id range nests inside another file's is
    caught by the probe_file stamp, not by prepare_chat_entries (audit List-1 #5)."""
    import json
    import judge_common
    primary = tmp_path / "probe_dataset_test_v2.json"
    primary.write_text("[]")
    diag_sc = tmp_path / "probe_context_overrides_test_v2_diag.json"
    diag_sc.write_text(json.dumps({"probe_file": "probe_dataset_test_v2_diag.json",
                                   "overrides": {"0": "ctx"}}))
    with pytest.raises(ValueError, match="wrong side-car"):
        judge_common.load_context_overrides(primary, explicit_path=diag_sc)
