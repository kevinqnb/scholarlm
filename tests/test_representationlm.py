"""Unit tests for RepresentationLM's key-term matching and token selection.

Rung 1 of the staged-gate ladder (see
``notes/scholarlm/builds/2026-08-31-representation-lm-01.md``): the
model-free, GPU-free logic — whole-word / plural / case matching and
last-subword token selection — on a tiny hand-built fixture whose output is
verifiable by inspection. The forward-pass rungs (2/3) need a GPU and run as
a submitted job, not here.

``select_last_subword_index`` is exercised against the real Llama-3.1-8B
tokenizer (skipped if it can't be loaded — gated repo / offline cache), since
the whole point is that a multi-subword term collects its *last* subword.
"""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scholarlm.representationlm import (
    find_key_term_occurrences,
    select_last_subword_index,
)

KEY_TERMS = ["pond", "lake", "wetland"]


# ---------------------------------------------------------------------------
# find_key_term_occurrences — pure regex matching
# ---------------------------------------------------------------------------


def test_repeated_term_one_row_per_occurrence():
    occ = find_key_term_occurrences("The pond near the other pond.", ["pond"])
    assert [t for t, _, _ in occ] == ["pond", "pond"]
    assert len(occ) == 2


def test_plural_matches_and_keeps_base_label():
    text = "Several ponds and one lake."
    occ = find_key_term_occurrences(text, ["pond", "lake"])
    assert occ == [("pond", 8, 13), ("lake", 22, 26)]
    assert text[8:13] == "ponds"        # plural surface form
    assert text[22:26] == "lake"


def test_whole_word_negatives():
    # 'pondweed' and 'respond' contain "pond" but are not whole-word matches.
    assert find_key_term_occurrences("pondweed respond correspond", ["pond"]) == []


def test_case_insensitive():
    occ = find_key_term_occurrences("Pond POND pond PoNd", ["pond"])
    assert len(occ) == 4
    assert {t for t, _, _ in occ} == {"pond"}


def test_multi_term_and_sorted_by_span():
    text = "a wetland, a lake, a pond, another wetland"
    occ = find_key_term_occurrences(text, KEY_TERMS)
    assert [t for t, _, _ in occ] == ["wetland", "lake", "pond", "wetland"]
    starts = [s for _, s, _ in occ]
    assert starts == sorted(starts)


def test_known_answer_total_row_count():
    # Tiny fixture; counts verifiable by inspection.
    docs = {
        "d0": "The pond and the lake.",                       # pond 1, lake 1
        "d1": "wetlands, wetland, WETLAND; pondweed",          # wetland 3, pond 0
        "d2": "no matches here at all",                        # 0
    }
    per_term = {t: 0 for t in KEY_TERMS}
    for text in docs.values():
        for term, _, _ in find_key_term_occurrences(text, KEY_TERMS):
            per_term[term] += 1
    assert per_term == {"pond": 1, "lake": 1, "wetland": 3}
    assert sum(per_term.values()) == 5


def test_empty_terms_and_bad_terms_raise():
    with pytest.raises(ValueError):
        find_key_term_occurrences("text", [])
    with pytest.raises(ValueError):
        find_key_term_occurrences("text", ["pond", " lake"])


# ---------------------------------------------------------------------------
# select_last_subword_index — against the real Llama-3.1-8B tokenizer
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def llama_tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.1-8B", cache_dir=os.environ.get("HF_CACHE")
        )
    except Exception as e:  # gated repo, no network, empty cache
        pytest.skip(f"Llama-3.1-8B tokenizer unavailable: {e}")


def _offsets(tokenizer, text):
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=True)
    return list(enc["input_ids"]), [tuple(o) for o in enc["offset_mapping"]]


def test_single_subword_term_selects_its_token(llama_tokenizer):
    text = "measured in the pond during summer"
    ids, offs = _offsets(llama_tokenizer, text)
    (occ,) = find_key_term_occurrences(text, ["pond"])
    _, ms, me = occ
    idx = select_last_subword_index(offs, ms, me)
    s, e = offs[idx]
    assert s <= me - 1 < e
    # The chosen token's text ends with the surface form.
    assert text[s:e].strip().endswith("pond")


def test_multi_subword_term_selects_last_subword(llama_tokenizer):
    text = "the wetlands near the shore"
    ids, offs = _offsets(llama_tokenizer, text)
    (occ,) = find_key_term_occurrences(text, ["wetland"])
    _, ms, me = occ
    assert text[ms:me] == "wetlands"

    # More than one token overlaps the match span (it is split).
    overlapping = [i for i, (s, e) in enumerate(offs) if s < me and e > ms]
    assert len(overlapping) >= 2

    idx = select_last_subword_index(offs, ms, me)
    # We picked the LAST of the overlapping subword tokens, not the first.
    assert idx == max(overlapping)
    s, e = offs[idx]
    assert s <= me - 1 < e


def test_plural_multi_subword_last_char(llama_tokenizer):
    text = "two lakes were sampled"
    ids, offs = _offsets(llama_tokenizer, text)
    (occ,) = find_key_term_occurrences(text, ["lake"])
    _, ms, me = occ
    assert text[ms:me] == "lakes"
    idx = select_last_subword_index(offs, ms, me)
    s, e = offs[idx]
    assert s <= me - 1 < e  # token contains the final 's'


def test_bos_token_never_selected(llama_tokenizer):
    text = "pond"
    ids, offs = _offsets(llama_tokenizer, text)
    # BOS is prepended with a (0, 0) offset.
    assert offs[0] == (0, 0)
    (occ,) = find_key_term_occurrences(text, ["pond"])
    _, ms, me = occ
    idx = select_last_subword_index(offs, ms, me)
    assert idx != 0
