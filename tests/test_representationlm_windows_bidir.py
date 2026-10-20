"""Rung-1 tests for RepresentationLM's cleaning, paragraph windows, mean pooling
and bidirectional (HF) backend.

Everything is offline: the "model" is a tiny randomly-initialised ModernBERT
saved to ``tmp_path`` with a tiny WordPiece tokenizer, so expected values are
checked against independent re-computation (manual means, a separate forward
on the window text), not against remembered numbers.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scholarlm.representationlm import (
    CleaningConfig,
    RepresentationLM,
    clean_ocr_text,
    paragraph_of_span,
    pool_rows,
    select_last_subword_index,
    select_term_token_indices,
    split_paragraphs,
    window_span,
)

KEY_TERMS = ["pond", "lake", "wetland"]

# ---------------------------------------------------------------------------
# cleaning
# ---------------------------------------------------------------------------

RAW = (
    '<page number="0">\n\nIntro about the pond here.\n\n'
    '<table number="1">\n<tr><td>lake</td></tr>\n</table>\n\n'
    'Second paragraph.\n\n</page>\n\n'
    '<page number="1">\n\nREFERENCES\n\n1. Smith, lake study.\n\n</page>\n'
)


def test_clean_all_removes_refs_tables_tags():
    out, info = clean_ocr_text(RAW, CleaningConfig())
    assert out == "Intro about the pond here.\n\nSecond paragraph."
    assert info["reference_heading_found"] is True
    assert info["n_tables_removed"] == 1
    assert info["n_page_tags_removed"] >= 2
    assert "lake" not in out  # both lake matches sat in removed spans


def test_clean_is_idempotent():
    once, _ = clean_ocr_text(RAW, CleaningConfig())
    twice, _ = clean_ocr_text(once, CleaningConfig())
    assert once == twice


def test_clean_all_off_is_untouched():
    out, info = clean_ocr_text(RAW, CleaningConfig(False, False, False))
    assert out == RAW and info["reference_heading_found"] is None


def test_clean_each_flag_independent():
    out, _ = clean_ocr_text(RAW, CleaningConfig(False, True, False))
    assert "<table" not in out and "<page" in out and "REFERENCES" in out
    out, _ = clean_ocr_text(RAW, CleaningConfig(False, False, True))
    assert "<page" not in out and "<table" in out


def test_clean_removed_span_never_glues_words():
    out, _ = clean_ocr_text("a<page number=\"0\">b", CleaningConfig(False, False, True))
    assert out == "a\n\nb"


def test_clean_no_reference_heading_is_reported():
    _, info = clean_ocr_text('<page number="0">\n\nno bib here\n\n</page>\n', CleaningConfig())
    assert info["reference_heading_found"] is False


TABLE_HEADER_DOC = (
    '<page number="0">\n\nIntro pond text here.\n\n'
    '<table number="1">\n<tr><th>Mean</th><th>References</th></tr>\n<tr><td>1</td><td>Bell</td></tr>\n</table>\n\n'
    'Body about the lake continues.\n\n</page>\n\n'
    '<page number="1">\n\nREFERENCES\n\n1. Smith.\n\n</page>\n'
)


def test_table_header_named_references_is_not_the_bibliography():
    out, info = clean_ocr_text(TABLE_HEADER_DOC, CleaningConfig())
    assert out == "Intro pond text here.\n\nBody about the lake continues."
    assert info["reference_heading_found"] is True
    assert info["kept_fraction_after_references"] < 1.0


def test_kept_fraction_guard_raises_on_early_cut():
    jstor = ('<page number="0">\n\nTitle\n\nREFERENCES\nLinked references are available on JSTOR\n\n'
             + "body words " * 200 + '\n\n</page>\n')
    out, info = clean_ocr_text(jstor, CleaningConfig())      # no guard: silently cut
    assert info["kept_fraction_after_references"] < 0.05 and "body words" not in out
    with pytest.raises(ValueError, match="min_kept_fraction_after_references=0.5"):
        clean_ocr_text(jstor, CleaningConfig(min_kept_fraction_after_references=0.5))
    # a normal end-of-paper bibliography passes the same guard
    clean_ocr_text(RAW, CleaningConfig(min_kept_fraction_after_references=0.3))


def test_guard_config_validation_and_doc_id_in_error():
    with pytest.raises(ValueError):
        CleaningConfig(min_kept_fraction_after_references=0)
    with pytest.raises(ValueError):
        CleaningConfig(remove_references=False, min_kept_fraction_after_references=0.5)


def test_cleaning_removes_exactly_the_matches_in_removed_spans():
    from scholarlm.representationlm import find_key_term_occurrences as f
    assert len(f(RAW, KEY_TERMS)) == 3          # pond, table lake, reference lake
    assert len(f(clean_ocr_text(RAW, CleaningConfig())[0], KEY_TERMS)) == 1


# ---------------------------------------------------------------------------
# paragraphs / windows / pooling helpers
# ---------------------------------------------------------------------------

TEXT = "p0 a.\n\np1 pond.\n\n \n\np2 b.\n\np3 c."


def test_split_paragraphs_and_window():
    paras = split_paragraphs(TEXT)
    assert [TEXT[s:e] for s, e in paras] == ["p0 a.", "p1 pond.", "p2 b.", "p3 c."]
    ms = TEXT.index("pond")
    assert paragraph_of_span(paras, ms, ms + 4) == 1
    a, b = window_span(paras, 1, 1)
    assert TEXT[a:b] == "p0 a.\n\np1 pond.\n\n \n\np2 b."
    assert window_span(paras, 0, 1) == (0, paras[1][1])        # clipped at start
    assert window_span(paras, 3, 1) == (paras[2][0], len(TEXT))  # clipped at end
    assert window_span(paras, 1, 0) == paras[1]


def test_paragraph_of_span_rejects_separator_span():
    paras = split_paragraphs(TEXT)
    with pytest.raises(AssertionError):
        paragraph_of_span(paras, paras[0][1], paras[1][0] + 1)


def test_select_term_tokens_and_last_agree():
    #            CLS     wet     ##land    ##s      SEP
    offs = [(0, 0), (4, 7), (7, 11), (11, 12), (0, 0)]
    assert select_term_token_indices(offs, 4, 12) == [1, 2, 3]
    assert select_last_subword_index(offs, 4, 12) == 3
    # space-prefixed token interval (3, 7) still overlaps match starting at 4
    assert select_term_token_indices([(0, 0), (3, 7), (0, 0)], 4, 7) == [1]


def test_pool_rows_known_answer():
    rows = np.array([[1, 2], [3, 4], [5, 6], [10, 10]], dtype=np.float32)
    out = pool_rows(rows, np.array([2, 1, 1]))
    np.testing.assert_array_equal(out, np.array([[2, 3], [5, 6], [10, 10]], dtype=np.float32))
    assert out.dtype == np.float32


# ---------------------------------------------------------------------------
# bidirectional backend on a tiny random ModernBERT
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    d = tmp_path_factory.mktemp("tiny_modernbert")
    words = ("the pond lake wet ##land ##s is near a and in water was cold deep big small "
             "old . , x y z study field").split()
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + words
    (d / "vocab.txt").write_text("\n".join(vocab))
    tok = transformers.BertTokenizerFast(str(d / "vocab.txt"), do_lower_case=True)
    tok.save_pretrained(d)
    cfg = transformers.ModernBertConfig(
        vocab_size=len(vocab), hidden_size=32, intermediate_size=64,
        num_hidden_layers=3, num_attention_heads=4, max_position_embeddings=40,
        local_attention=8, global_attn_every_n_layers=3,
        pad_token_id=0, cls_token_id=2, sep_token_id=3, bos_token_id=2, eos_token_id=3,
    )
    torch.manual_seed(0)
    transformers.ModernBertModel(cfg).save_pretrained(d)
    return str(d)


def _lm(path, **kw):
    args = dict(architecture="bidirectional", layers=[0, 1, 2, 3], device="cpu")
    args.update(kw)
    return RepresentationLM(path, **args)


DOC = ("the water was cold in the pond .\n\n"
       "a big lake is near the deep pond and the wetlands .\n\n"
       "the old study field was small .\n\n"
       "x y z .")


def test_constructor_guards(tiny_model_dir):
    with pytest.raises(ValueError, match="explicit `device`"):
        RepresentationLM(tiny_model_dir, architecture="bidirectional", layers=[1], device=None)
    with pytest.raises(ValueError, match="tail_drop is causal-only"):
        _lm(tiny_model_dir, overflow_policy="tail_drop")
    with pytest.raises(ValueError, match="architecture"):
        RepresentationLM(tiny_model_dir, architecture="auto", layers=[1], device="cpu")
    with pytest.raises(ValueError, match="layer 9 out of range"):
        _lm(tiny_model_dir, layers=[9])


def test_unvalidated_model_type_rejected(tiny_model_dir, tmp_path):
    import json, shutil
    d = tmp_path / "fake_llm2vec"
    shutil.copytree(tiny_model_dir, d)
    cfg = json.loads((d / "config.json").read_text())
    cfg["model_type"] = "llama"
    (d / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="LLM2Vec is not supported"):
        _lm(str(d))


def test_mean_vs_manual_and_single_token_equals_last(tiny_model_dir):
    mean = _lm(tiny_model_dir, pooling="mean", cleaning=None)
    last = _lm(tiny_model_dir, pooling="last", cleaning=None)
    om = mean.collect({"d": DOC}, KEY_TERMS)
    ol = last.collect({"d": DOC}, KEY_TERMS)
    assert [DOC[s:e] for s, e in zip(om["char_starts"], om["char_ends"])] == [
        "pond", "lake", "pond", "wetlands"]
    assert om["n_pooled_tokens"].tolist() == [1, 1, 1, 3]     # wet ##land ##s
    assert (ol["n_pooled_tokens"] == 1).all()
    np.testing.assert_array_equal(om["token_indices"], ol["token_indices"])
    for L in mean.layers:
        # single-token terms: bitwise equal
        np.testing.assert_array_equal(om["representations"][L][:3], ol["representations"][L][:3])
        # multi-token: mean of the three per-token states, recomputed independently
        ids, offs = mean._tokenize(DOC)
        ms = DOC.index("wetlands")
        toks = select_term_token_indices(offs, ms, ms + 8)
        out = mean._forward_bidirectional(ids)
        h = mean._layer_state(out, L)[toks].float().numpy()
        np.testing.assert_allclose(om["representations"][L][3], h.mean(0), rtol=1e-6, atol=1e-6)
        assert not np.allclose(om["representations"][L][3], ol["representations"][L][3])


def test_window_rows_equal_separate_forward_on_window_text(tiny_model_dir):
    lm = _lm(tiny_model_dir, context_radius=1, pooling="mean", cleaning=None)
    out = lm.collect({"d": DOC}, KEY_TERMS)
    paras = split_paragraphs(DOC)
    # pond(p0) -> window p0..p1 ; lake, pond, wetlands (p1) -> window p0..p2
    assert out["paragraph_indices"].tolist() == [0, 1, 1, 1]
    a0, b0 = window_span(paras, 0, 1)
    a1, b1 = window_span(paras, 1, 1)
    assert out["window_starts"].tolist() == [a0, a1, a1, a1]
    assert out["window_ends"].tolist() == [b0, b1, b1, b1]
    whole = _lm(tiny_model_dir, context_radius=None, cleaning=None)
    ref = whole.collect({"w": DOC[a1:b1]}, KEY_TERMS)  # same window as a standalone doc
    for L in lm.layers:
        np.testing.assert_array_equal(out["representations"][L][1:], ref["representations"][L][1:])
    # window context differs from full-doc context (ablation direction)
    full = whole.collect({"d": DOC}, KEY_TERMS)
    assert not np.allclose(out["representations"][3][1:], full["representations"][3][1:])


def test_radius_zero_uses_only_own_paragraph(tiny_model_dir):
    lm = _lm(tiny_model_dir, context_radius=0, cleaning=None)
    out = lm.collect({"d": DOC}, KEY_TERMS)
    paras = split_paragraphs(DOC)
    assert out["window_starts"].tolist() == [paras[0][0]] + [paras[1][0]] * 3


def test_cleaning_changes_rows_and_offsets_index_cleaned_text(tiny_model_dir):
    raw = '<page number="0">\n\n' + DOC + '\n\n</page>\n\n<page number="1">\n\nREFERENCES\n\na lake .\n\n</page>\n'
    lm = _lm(tiny_model_dir, cleaning=CleaningConfig())
    cleaned, info = lm.prepare_documents({"d": raw})
    assert cleaned["d"] == DOC and info["d"]["reference_heading_found"] is True
    out = lm.collect({"d": raw}, KEY_TERMS)
    assert [DOC[s:e] for s, e in zip(out["char_starts"], out["char_ends"])] == [
        "pond", "lake", "pond", "wetlands"]          # reference-section lake gone
    assert lm.doc_info["d"]["cleaned_sha256"] == info["d"]["cleaned_sha256"]


def test_overflow_raises(tiny_model_dir):
    long_doc = " ".join(["the pond is deep ."] * 30)   # >> 40 tokens, one paragraph
    with pytest.raises(ValueError, match="max_position_embeddings"):
        _lm(tiny_model_dir, cleaning=None).collect({"d": long_doc}, KEY_TERMS)


def test_seed_determinism(tiny_model_dir):
    lm = _lm(tiny_model_dir, context_radius=1)
    a = lm.collect({"d": DOC}, KEY_TERMS)
    b = lm.collect({"d": DOC}, KEY_TERMS)
    for L in lm.layers:
        np.testing.assert_array_equal(a["representations"][L], b["representations"][L])


def test_verify_read_point_bidirectional_passes(tiny_model_dir, capsys):
    _lm(tiny_model_dir, cleaning=None).verify_read_point(DOC, KEY_TERMS)
    assert "future-dependence check OK" in capsys.readouterr().out


def test_verify_read_point_rejects_overlength_sample(tiny_model_dir):
    long_doc = " ".join(["the pond is deep ."] * 30)
    with pytest.raises(ValueError, match="verify_read_point: sample is"):
        _lm(tiny_model_dir, cleaning=None).verify_read_point(long_doc, KEY_TERMS)


def test_verify_read_point_windows_to_first_term(tiny_model_dir, capsys):
    # whole doc is > 40 tokens but the radius-0 window around the first term fits
    doc = "the pond was cold .\n\n" + " ".join(["the lake is deep ."] * 30)
    _lm(tiny_model_dir, cleaning=None, context_radius=0).verify_read_point(doc, KEY_TERMS)
    assert "future-dependence check OK" in capsys.readouterr().out
