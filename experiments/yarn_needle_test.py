"""One-off needle-in-a-haystack diagnostic for the YaRN rope_scaling override in
experiments/model-configs/interp_judge/qwen-2.5-7b.yaml.

NOT part of the judge pipeline -- standalone, no NNsight, no JudgementLM true/false
template. The 2026-09-16 smoke test (2026-09-16-supermat-qwen-2.5-7b-yarn-smoke-
test-01) ran the real judge task on a real long document and got a wrong, confident
verdict, but that conflates several variables (judge-template calibration, domain
reasoning about the entity, true/false answer-cue mechanics) with the one question
that actually matters: can this model, with this rope_scaling config, retrieve
*anything* from the tail of a document past its native 32768-token window. This
script isolates exactly that with plain generation.

Plants a unique marker sentence in supermat's longest OCR document
(hott2013review-CC.txt, measured ~68k tok) at two positions:
  - "near" control: within the native window. If the model can't retrieve THIS,
    something more basic than RoPE is broken (chat template, tokenizer, generation
    config) and the far result below is uninterpretable.
  - "far" test: appended at the tail of the FULL ~68k-token document, past the
    native window -- this is what YaRN is supposed to fix.

Usage: python experiments/yarn_needle_test.py
"""
from __future__ import annotations

from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

_REPO_ROOT = Path(__file__).parent.parent
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_CONFIG_PATH = _REPO_ROOT / "experiments/model-configs/interp_judge/qwen-2.5-7b.yaml"
DOC_PATH = _REPO_ROOT / "data/supermat/ocr_output_raw/hott2013review-CC.txt"
MARKER_PHRASE = "zebra-ionosphere-77"
MARKER = f"The secret verification passphrase is: {MARKER_PHRASE}."


def ask(model, tok, context: str, question: str, label: str) -> bool:
    messages = [
        {"role": "system", "content": "Answer with ONLY the exact passphrase found "
                                       "in the document. No other words, no punctuation."},
        {"role": "user", "content": f"{context}\n\n{question}"},
    ]
    input_ids = tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    n_tokens = input_ids.shape[1]
    with torch.no_grad():
        out = model.generate(input_ids, max_new_tokens=20, do_sample=False)
    answer = tok.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)
    hit = MARKER_PHRASE in answer
    print(f"[{label}] prompt_tokens={n_tokens} answer={answer!r} FOUND={hit}")
    return hit


def main() -> None:
    with open(MODEL_CONFIG_PATH) as f:
        judge_cfg = yaml.safe_load(f)
    nnsight_kwargs = judge_cfg["nnsight_kwargs"]
    rope_scaling = nnsight_kwargs["rope_scaling"]
    torch_dtype = getattr(torch, nnsight_kwargs["torch_dtype"])

    print(f"Loading {MODEL_ID}, rope_scaling={rope_scaling}, torch_dtype={torch_dtype}")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch_dtype, rope_scaling=rope_scaling, device_map="cuda",
    )

    doc = DOC_PATH.read_text()
    question = "What is the exact passphrase written in the document above?"

    # Near control: marker inserted a few thousand words in, well inside the
    # native 32768-token window, using only a short prefix of the document (not
    # the whole thing) so this call is unambiguously a short-context baseline.
    words = doc.split()
    near_doc = " ".join(words[:2000]) + f"\n\n{MARKER}\n\n" + " ".join(words[2000:8000])
    near_hit = ask(model, tok, near_doc, question, "near-control (short context)")

    # Far test: the FULL document, marker appended at the very tail -- past the
    # native window by construction (this file alone measured ~68k tok).
    far_doc = doc + f"\n\n{MARKER}\n"
    far_hit = ask(model, tok, far_doc, question, "far-test (full doc, marker at tail)")

    print()
    print(f"near-control FOUND={near_hit}  far-test FOUND={far_hit}")
    if not near_hit:
        print("near-control failed -- something more basic than RoPE is broken; "
              "far-test result is uninterpretable until this passes.")
    elif near_hit and not far_hit:
        print("near-control passed, far-test failed -- consistent with the "
              "rope_scaling override not actually extending usable attention range.")
    elif near_hit and far_hit:
        print("both passed -- the model CAN retrieve tail content at this length; "
              "the smoke test's wrong verdict is likely a judge-task/semantic issue, "
              "not a long-context retrieval failure.")


if __name__ == "__main__":
    main()
