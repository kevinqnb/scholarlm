"""Sentence segmentation for the ChatExtract baseline.

ChatExtract (Polak & Morgan, Nat. Commun. 2024) operates one sentence at a
time: OCR prose is split into individual sentences, each classified and then
extracted from within the context of a short surrounding passage. This module
isolates that sentence-splitting step behind a single function so the
`pysbd` dependency stays contained and testable.

`pysbd` (Python Sentence Boundary Disambiguation) is used rather than a naive
period split because scientific prose is dense with decimals ("6.9", "1.8 m"),
abbreviations ("Fig. 2", "e.g.", "et al."), and unit tokens that a regex split
on "." would shred — corrupting exactly the numeric values ChatExtract is
trying to read. `pysbd` is pure-Python and needs no model download.
"""

from __future__ import annotations

import pysbd

# A single reusable segmenter. `clean=False` keeps the original text intact
# (no whitespace/entity normalization) so downstream passages match the source.
_SEGMENTER = pysbd.Segmenter(language="en", clean=False)


def split_sentences(text: str) -> list[str]:
    """Split a block of prose into sentences.

    Returns a list of non-empty, stripped sentence strings. Newlines within a
    sentence (common in OCR output where a single sentence wraps across lines)
    are collapsed to spaces so each returned sentence is a single clean line.
    """
    if not text or not text.strip():
        return []
    sentences = []
    for sent in _SEGMENTER.segment(text):
        cleaned = " ".join(sent.split())
        if cleaned:
            sentences.append(cleaned)
    return sentences
