"""
Compute a confidence score in [0, 1] for every extracted measurement with a vLLM-served judge.

Methods (``params.baseline``):
  - ``verbalized``: the judge states an integer confidence from 0 to 100; score = value / 100.
  - ``sampling``: the judge answers true/false ``n_samples`` times; score = fraction true.
  - ``ntp``: score = probability of "true" relative to "false" as the next token.

All methods send the same single user message (instructions, paper, query), except that
``verbalized`` asks for a number. A row whose answers stay unusable after its retries is
marked failed with empty output; any other error, such as a failed request, aborts the run.

Config params (all required unless marked optional):
  - all methods: dataset, judge, baseline, ocr_dir, and one of extraction_id / synthetic_file;
    optional: walltime, api_base, row_indices (rows of the input to score; default all)
  - verbalized: max_tokens, max_retries, sampling_params, retry_temperature
  - sampling: max_tokens, max_retries, sampling_params, n_samples
  - ntp: top_logprobs
  ``sampling_params`` holds temperature, top_p, top_k, min_p and repetition_penalty.

Output (experiments/results/{dataset}/judge_baselines/{id}/):
  - ``responses.json``: input rows plus judgement_p_true (the score), judgement_raw_text,
    judgement_model and judgement_failed; the first three are null on a failed row
  - ``raw_outputs.json``: per-row detail by input row index, also written on abort
  - ``run_metadata.json``

Usage:
    python experiments/run_judge_baselines.py <config>
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).parent.parent
_EXPERIMENTS_DIR = Path(__file__).parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_EXPERIMENTS_DIR))

from dotenv import load_dotenv
load_dotenv()

from openai import AsyncOpenAI

from scholarlm.config import DatasetConfig
from scholarlm.instruction_prompts import JUDGE_CONFIDENCE_INSTRUCTIONS, JUDGE_INSTRUCTIONS
from scholarlm.judgementlm import tokenize
from scholarlm.utils import judge_prompts

from run_extraction import load_dataset_config
import utils as paths
from utils import set_seeds, write_run_metadata

# Params each method requires, beyond the ones every method takes.
METHOD_PARAMS: dict[str, set[str]] = {
    "verbalized": {"max_tokens", "max_retries", "sampling_params", "retry_temperature"},
    "sampling": {"max_tokens", "max_retries", "sampling_params", "n_samples"},
    "ntp": {"top_logprobs"},
}
COMMON_PARAMS = {"dataset", "judge", "baseline", "ocr_dir"}
INPUT_PARAMS = {"extraction_id", "synthetic_file"}
OPTIONAL_PARAMS = {"walltime", "api_base", "row_indices"}

# vLLM fills any of these that a request omits from the model's generation_config.json,
# so all of them must be set explicitly.
SAMPLING_KEYS = {"temperature", "top_p", "top_k", "min_p", "repetition_penalty"}

# Closing line of the true/false judge query, and its confidence counterpart.
VERDICT_CLOSING = "(true or false) Is this extraction correct?"
CONFIDENCE_CLOSING = "(0 to 100) How confident are you that this extraction is correct?"

LABELS = ("true", "True", "false", "False")

OUTPUT_FIELDS = ("judgement_p_true", "judgement_raw_text", "judgement_model", "judgement_failed")


class RowFailure(Exception):
    """A row whose answers stayed unusable after all retries.

    Attributes:
        detail: JSON-serializable answers to keep for inspection.
    """

    def __init__(self, message: str, detail: Any) -> None:
        """Create a failure.

        Args:
            message: Description of what went wrong.
            detail: JSON-serializable answers to keep for inspection.
        """
        super().__init__(message)
        self.detail = detail


# ---------------------------------------------------------------------------
# Parsing, prompts, seeds
# ---------------------------------------------------------------------------


def parse_verdict(text: str) -> bool | None:
    """Read a true/false verdict from a judge answer.

    Args:
        text: The judge's answer, stripped of surrounding whitespace.

    Returns:
        True or False if the answer's first word is "true" or "false" (any casing, leading
        punctuation ignored), otherwise None.
    """
    m = re.match(r"\W*(true|false)\b", text, re.IGNORECASE)
    return None if m is None else m.group(1).lower() == "true"


def parse_confidence(text: str) -> float | None:
    """Read a 0-100 integer confidence from a judge answer.

    Out-of-range values are not clipped. Decimals are rejected because "0.85" would
    otherwise be scored as 0.0085.

    Args:
        text: The judge's answer, stripped of surrounding whitespace.

    Returns:
        The stated value / 100, in [0, 1]. None unless the answer's first line is one
        integer from 0 to 100 (a trailing "%" or "." is allowed).
    """
    m = re.match(r"(\d{1,3})[ \t]*%?\.?[ \t]*(?:\n|$)", text)
    if m is None:
        return None
    value = int(m.group(1))
    return value / 100 if value <= 100 else None


def to_confidence_entries(entries: list[dict], dataset_config: DatasetConfig) -> list[dict]:
    """Convert true/false chat entries into 0-100 confidence entries.

    Replaces the instructions and the query's closing line; the paper text and the
    description of the extraction are unchanged.

    Args:
        entries: Chat entries from ``judge_prompts.prepare_chat_entries``. Reads keys
            ``custom_id`` and ``user_query``.
        dataset_config: Configuration of the dataset the entries came from.

    Returns:
        New entries, in the same order, with ``system`` and ``user_query`` replaced.

    Raises:
        ValueError: If the dataset defines its own judge prompt, or a query does not end
            with the expected closing line.
    """
    if dataset_config.judge_instructions not in (None, JUDGE_INSTRUCTIONS):
        raise ValueError(
            f"{dataset_config.name}: custom judge_instructions have no confidence variant"
        )
    out = []
    for e in entries:
        query = e["user_query"]
        if not query.endswith(VERDICT_CLOSING):
            raise ValueError(f"[idx={e['custom_id']}] unexpected judge query layout")
        out.append(e | {
            "system": JUDGE_CONFIDENCE_INSTRUCTIONS,
            "user_query": query[: -len(VERDICT_CLOSING)] + CONFIDENCE_CLOSING,
        })
    return out


def user_message(entry: dict) -> str:
    """Build the single user message sent for a chat entry.

    Args:
        entry: Chat entry; reads ``system`` (the instructions), ``context_text`` (the
            paper) and ``user_query``.

    Returns:
        The message text, with INSTRUCTIONS, CONTEXT and QUERY sections.
    """
    return (
        f"## INSTRUCTIONS:\n{entry['system']}\n\n"
        f"## CONTEXT:\n{entry['context_text']}\n\n"
        f"## QUERY:\n{entry['user_query']}"
    )


def sample_seed(base_seed: int, idx: int, sample: int, attempt: int) -> int:
    """Derive the sampling seed of one request.

    Every (row, sample, attempt) gets its own seed. A seed shared across samples would
    make a row's samples identical; one shared across rows would make the rows' random
    errors correlated.

    Args:
        base_seed: The experiment seed.
        idx: Row index in the input file.
        sample: Index of the sample within the row.
        attempt: 0 for the first request, 1 for the first retry, and so on.

    Returns:
        An integer in [0, 2**31).
    """
    digest = hashlib.sha256(f"{base_seed}:{idx}:{sample}:{attempt}".encode()).digest()
    return int.from_bytes(digest[:4], "big") >> 1


# ---------------------------------------------------------------------------
# Verbalized confidence and repeated sampling
# ---------------------------------------------------------------------------


async def ask(
    client: AsyncOpenAI,
    model_id: str,
    entry: dict,
    idx: int,
    sem: asyncio.Semaphore,
    parse: Callable[[str], Any],
    *,
    sample: int,
    base_seed: int,
    sampling_params: dict[str, float],
    retry_temperature: float,
    max_tokens: int,
    max_retries: int,
    reasoning_effort: str | None,
) -> dict:
    """Ask the judge one question, retrying while its answer cannot be parsed.

    Args:
        client: Client connected to the vLLM server.
        model_id: Model name the server was launched with.
        entry: Chat entry, sent as the single message built by ``user_message``.
        idx: Row index in the input file, used for the seed and in messages.
        sem: Limits the number of requests in flight.
        parse: Maps the answer text to its value, or None if it is unusable.
        sample: Index of this sample within the row, used for the seed.
        base_seed: The experiment seed.
        sampling_params: Value for every key in ``SAMPLING_KEYS``.
        retry_temperature: Temperature used for retries in place of
            ``sampling_params["temperature"]``.
        max_tokens: Maximum number of generated tokens.
        max_retries: Number of additional requests allowed after an unusable answer.
        reasoning_effort: Passed to the chat template if not None.

    Returns:
        Dict with ``text`` (the accepted answer), ``parsed`` (its value), ``rejected``
        (the unusable answers before it), ``model`` (model name reported by the server)
        and ``prompt_tokens``.

    Raises:
        RowFailure: If no answer can be parsed within ``1 + max_retries`` requests.
        openai.APIError: If a request fails.
    """
    extra_body: dict[str, Any] = {k: sampling_params[k] for k in ("top_k", "min_p", "repetition_penalty")}
    if reasoning_effort is not None:
        extra_body["chat_template_kwargs"] = {"reasoning_effort": reasoning_effort}

    rejected: list[str] = []
    for attempt in range(1 + max_retries):
        async with sem:
            response = await client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": user_message(entry)}],
                max_tokens=max_tokens,
                temperature=sampling_params["temperature"] if attempt == 0 else retry_temperature,
                top_p=sampling_params["top_p"],
                seed=sample_seed(base_seed, idx, sample, attempt),
                extra_body=extra_body,
            )

        choice = response.choices[0]
        text = (choice.message.content or "").strip()
        parsed = parse(text)
        if parsed is not None:
            return {
                "text": text,
                "parsed": parsed,
                "rejected": rejected,
                "model": response.model,
                "prompt_tokens": response.usage.prompt_tokens,
            }
        rejected.append(text)
        print(f"  [idx={idx}, sample={sample}] unusable answer {attempt + 1}/{1 + max_retries} "
              f"(finish_reason={choice.finish_reason!r}): {text!r}")

    raise RowFailure(
        f"[idx={idx}, sample={sample}] no usable answer in {1 + max_retries} attempts",
        {"rejected": rejected},
    )


async def score_verbalized(
    client: AsyncOpenAI,
    model_id: str,
    entry: dict,
    idx: int,
    sem: asyncio.Semaphore,
    *,
    base_seed: int,
    sampling_params: dict[str, float],
    retry_temperature: float,
    max_tokens: int,
    max_retries: int,
    reasoning_effort: str | None,
) -> dict:
    """Score one row by the confidence the judge states.

    Args:
        client: Client connected to the vLLM server.
        model_id: Model name the server was launched with.
        entry: Confidence chat entry from ``to_confidence_entries``.
        idx: Row index in the input file.
        sem: Limits the number of requests in flight.
        base_seed: The experiment seed.
        sampling_params: Value for every key in ``SAMPLING_KEYS``.
        retry_temperature: Temperature used for retries.
        max_tokens: Maximum number of generated tokens.
        max_retries: Number of retries allowed after an unusable answer.
        reasoning_effort: Passed to the chat template if not None.

    Returns:
        Dict with ``score`` (stated confidence / 100), ``raw_text`` (the answer),
        ``rejected`` (unusable answers before it), ``model`` and ``prompt_tokens``.

    Raises:
        RowFailure: If the retries are exhausted.
        openai.APIError: If a request fails.
    """
    r = await ask(
        client, model_id, entry, idx, sem, parse_confidence,
        sample=0, base_seed=base_seed, sampling_params=sampling_params,
        retry_temperature=retry_temperature, max_tokens=max_tokens, max_retries=max_retries,
        reasoning_effort=reasoning_effort,
    )
    print(f"  [idx={idx}] Response: {r['text']}")
    return {
        "score": r["parsed"],
        "raw_text": r["text"],
        "rejected": r["rejected"],
        "model": r["model"],
        "prompt_tokens": r["prompt_tokens"],
    }


async def score_sampling(
    client: AsyncOpenAI,
    model_id: str,
    entry: dict,
    idx: int,
    sem: asyncio.Semaphore,
    *,
    n_samples: int,
    base_seed: int,
    sampling_params: dict[str, float],
    max_tokens: int,
    max_retries: int,
    reasoning_effort: str | None,
) -> dict:
    """Score one row by the fraction of sampled true/false answers that are true.

    Each sample is its own request with its own seed and its own retry budget, so the
    denominator is always ``n_samples``: one failed sample fails the row.

    Args:
        client: Client connected to the vLLM server.
        model_id: Model name the server was launched with.
        entry: True/false chat entry.
        idx: Row index in the input file.
        sem: Limits the number of requests in flight.
        n_samples: Number of answers to sample.
        base_seed: The experiment seed.
        sampling_params: Value for every key in ``SAMPLING_KEYS``; retries use the same
            temperature as first attempts.
        max_tokens: Maximum number of generated tokens per answer.
        max_retries: Number of retries allowed per sample.
        reasoning_effort: Passed to the chat template if not None.

    Returns:
        Dict with ``score`` (fraction of samples that are true), ``raw_text`` (list of
        the ``n_samples`` answers), ``rejected`` (unusable answers across all samples),
        ``n_nonliteral`` (number of answers that are not exactly "true" or "false" up to
        casing), ``model`` and ``prompt_tokens`` (prompt length of one request).

    Raises:
        RowFailure: If any sample exhausts its retries. The detail holds the outcome of
            every sample.
        openai.APIError: If a request fails.
    """
    async def one_sample(k: int) -> dict | RowFailure:
        """Draw one sample.

        Args:
            k: Index of the sample within the row.

        Returns:
            The result of ``ask``, or the failure if its retries are exhausted.
        """
        try:
            return await ask(
                client, model_id, entry, idx, sem, parse_verdict,
                sample=k, base_seed=base_seed, sampling_params=sampling_params,
                retry_temperature=sampling_params["temperature"], max_tokens=max_tokens,
                max_retries=max_retries, reasoning_effort=reasoning_effort,
            )
        except RowFailure as e:
            return e

    samples = await asyncio.gather(*[one_sample(k) for k in range(n_samples)])
    assert len(samples) == n_samples
    errors = [x for x in samples if isinstance(x, RowFailure)]
    if errors:
        raise RowFailure(
            f"[idx={idx}] {len(errors)}/{n_samples} samples failed; first: {errors[0]}",
            {"samples": [
                {"error": str(x), "detail": x.detail} if isinstance(x, RowFailure) else x["text"]
                for x in samples
            ]},
        )
    score = sum(s["parsed"] for s in samples) / n_samples
    print(f"  [idx={idx}] Score: {score}")
    return {
        "score": score,
        "raw_text": [s["text"] for s in samples],
        "rejected": [text for s in samples for text in s["rejected"]],
        "n_nonliteral": sum(s["text"].lower() not in ("true", "false") for s in samples),
        "model": samples[0]["model"],
        "prompt_tokens": samples[0]["prompt_tokens"],
    }


# ---------------------------------------------------------------------------
# Next-token probability
# ---------------------------------------------------------------------------


def ntp_from_logprobs(logprobs: dict[str, float]) -> tuple[float, float, float]:
    """Compute relative true/false probabilities from next-token log-probabilities.

    With P the model's next-token distribution:
        p_true = (P("true") + P("True")) / label_mass
        label_mass = P("true") + P("True") + P("false") + P("False")
    Sums are taken in log space so that very small probabilities do not underflow.

    Args:
        logprobs: Natural-log probability of each label in ``LABELS``.

    Returns:
        Tuple ``(p_true, p_false, label_mass)``. ``p_true + p_false`` is 1; ``label_mass``
        is the total probability the model gives to the four labels.

    Raises:
        ValueError: If a log-probability is not finite.
    """
    def logaddexp(a: float, b: float) -> float:
        """Add two probabilities given in log space.

        Args:
            a: First log-probability.
            b: Second log-probability.

        Returns:
            log(exp(a) + exp(b)).
        """
        m = max(a, b)
        return m + math.log(math.exp(a - m) + math.exp(b - m))

    if not all(math.isfinite(v) for v in logprobs.values()):
        raise ValueError(f"non-finite label log-probability: {logprobs}")
    log_true = logaddexp(logprobs["true"], logprobs["True"])
    log_false = logaddexp(logprobs["false"], logprobs["False"])
    log_mass = logaddexp(log_true, log_false)
    return math.exp(log_true - log_mass), math.exp(log_false - log_mass), math.exp(log_mass)


def label_token_ids(tokenizer: Any) -> dict[str, int]:
    """Look up the token id of each label.

    Args:
        tokenizer: The judge model's HuggingFace tokenizer.

    Returns:
        Token id of each label in ``LABELS``.

    Raises:
        ValueError: If a label is not exactly one token, or two labels share a token.
    """
    ids = {}
    for label in LABELS:
        toks = tokenizer.encode(label, add_special_tokens=False)
        if len(toks) != 1:
            raise ValueError(f"label {label!r} is not a single token: {toks}")
        ids[label] = toks[0]
    if len(set(ids.values())) != len(LABELS):
        raise ValueError(f"label token ids are not distinct: {ids}")
    return ids


async def lookup_logprob(
    client: AsyncOpenAI, model_id: str, prompt_ids: list[int], token_id: int
) -> float:
    """Get the log-probability of one specific token as the next token.

    Uses vLLM's /generative_scoring endpoint, which reads any requested token from the
    full next-token distribution, whatever its rank. The endpoint truncates over-long
    prompts silently, so the prompt length it reports is checked.

    Args:
        client: Client connected to the vLLM server; its base URL must end in /v1.
        model_id: Model name the server was launched with.
        prompt_ids: Token ids of the full prompt.
        token_id: Token to look up.

    Returns:
        Natural-log probability of ``token_id`` following the prompt.

    Raises:
        ValueError: If the base URL, the reported prompt length or the returned
            probability is not as expected.
        openai.APIError: If the request fails.
    """
    base = str(client.base_url).rstrip("/")
    if not base.endswith("/v1"):
        raise ValueError(f"expected api_base ending in /v1, got {base!r}")
    # The endpoint scores the concatenation of `query` and `item`, so the prompt is split
    # before its last token. apply_softmax=False requests the full-vocabulary probability.
    resp = await client.post(
        base[: -len("/v1")] + "/generative_scoring",
        cast_to=object,
        body={
            "model": model_id,
            "query": prompt_ids[:-1],
            "items": [prompt_ids[-1:]],
            "label_token_ids": [token_id],
            "apply_softmax": False,
        },
    )
    if resp["usage"]["prompt_tokens"] != len(prompt_ids):
        raise ValueError(
            f"generative_scoring saw {resp['usage']['prompt_tokens']} prompt tokens, sent {len(prompt_ids)}",
        )
    (item,) = resp["data"]
    if not 0.0 < item["score"] <= 1.0 + 1e-6:
        raise ValueError(f"token probability out of (0, 1]: {item['score']}")
    return math.log(item["score"])


async def score_ntp(
    client: AsyncOpenAI,
    model_id: str,
    entry: dict,
    idx: int,
    sem: asyncio.Semaphore,
    *,
    tokenizer: Any,
    label_ids: dict[str, int],
    top_logprobs: int,
) -> dict:
    """Score one row by the next-token probability of true relative to false.

    The server returns the ``top_logprobs`` most likely next tokens, with probabilities
    computed before any sampling transformation. Labels outside that list are looked up
    one by one.

    Args:
        client: Client connected to the vLLM server.
        model_id: Model name the server was launched with.
        entry: True/false chat entry; ``system``, ``context_text`` and ``user_query`` are
            combined into a single user message.
        idx: Row index in the input file.
        sem: Limits the number of rows in flight.
        tokenizer: The judge model's HuggingFace tokenizer.
        label_ids: Token id of each label in ``LABELS``.
        top_logprobs: Number of most likely tokens to request.

    Returns:
        Dict with ``score`` (p_true), ``raw_text`` (the most likely next token),
        ``p_false``, ``label_mass``, ``label_logprobs`` (per label), ``labels_looked_up``
        (labels outside the top list), ``top_token_is_label`` (whether the most likely
        token is a label), ``model`` and ``prompt_tokens``.

    Raises:
        ValueError: If the server's response is not as expected.
        openai.APIError: If a request fails.
    """
    async with sem:
        # Tokenizing blocks the event loop, so only rows about to be sent are tokenized.
        prompt_ids = tokenize(entry["system"], entry["context_text"], entry["user_query"], tokenizer)[0]
        response = await client.completions.create(
            model=model_id,
            prompt=prompt_ids,
            max_tokens=1,
            temperature=0.0,
            logprobs=top_logprobs,
            extra_body={"return_tokens_as_token_ids": True},
        )
        if response.usage.prompt_tokens != len(prompt_ids):
            raise ValueError(
                f"[idx={idx}] server saw {response.usage.prompt_tokens} prompt tokens, sent {len(prompt_ids)}"
            )
        top = {
            int(token.removeprefix("token_id:")): lp
            for token, lp in response.choices[0].logprobs.top_logprobs[0].items()
        }
        logprobs = {label: top[tid] for label, tid in label_ids.items() if tid in top}
        looked_up = [label for label in LABELS if label not in logprobs]
        for label in looked_up:
            logprobs[label] = await lookup_logprob(client, model_id, prompt_ids, label_ids[label])

    top_id = max(top, key=top.get)
    p_true, p_false, label_mass = ntp_from_logprobs(logprobs)
    print(f"  [idx={idx}] p_true: {p_true:.6g} (label mass {label_mass:.4f})")
    return {
        "score": p_true,
        "raw_text": tokenizer.decode([top_id]),
        "p_false": p_false,
        "label_mass": label_mass,
        "label_logprobs": logprobs,
        "labels_looked_up": looked_up,
        "top_token_is_label": top_id in label_ids.values(),
        "model": response.model,
        "prompt_tokens": response.usage.prompt_tokens,
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def validate_params(params: dict, judge_cfg: dict, config_path: Path | str) -> None:
    """Check that an experiment's params are complete and consistent.

    Args:
        params: The experiment config's ``params`` mapping.
        judge_cfg: The judge's model-config.
        config_path: Path of the experiment config, used in error messages.

    Raises:
        ValueError: If a required param is missing, a param is unknown or does not apply
            to the chosen method, or a value is invalid.
    """
    def fail(message: str) -> None:
        """Raise a config error.

        Args:
            message: Description of the problem.

        Raises:
            ValueError: Always.
        """
        raise ValueError(f"{config_path}: {message}")

    paths.require_params(params, *sorted(COMMON_PARAMS), config_path=config_path)
    baseline = params["baseline"]
    if baseline not in METHOD_PARAMS:
        fail(f"params.baseline must be one of {sorted(METHOD_PARAMS)}, got {baseline!r}")
    paths.require_params(params, *sorted(METHOD_PARAMS[baseline]), config_path=config_path)
    if len(INPUT_PARAMS & set(params)) != 1:
        fail(f"set exactly one of {sorted(INPUT_PARAMS)}")
    unknown = set(params) - COMMON_PARAMS - INPUT_PARAMS - OPTIONAL_PARAMS - METHOD_PARAMS[baseline]
    if unknown:
        fail(f"params not used by baseline {baseline!r}: {sorted(unknown)}")

    for key, minimum in (("max_tokens", 1), ("max_retries", 0), ("n_samples", 2), ("top_logprobs", 1)):
        if key in params and (type(params[key]) is not int or params[key] < minimum):
            fail(f"params.{key} must be an int >= {minimum}, got {params[key]!r}")
    if "row_indices" in params:
        rows = params["row_indices"]
        if not isinstance(rows, list) or not rows or len(set(rows)) != len(rows) or any(
            type(i) is not int or i < 0 for i in rows
        ):
            fail("params.row_indices must be a non-empty list of distinct non-negative ints")
    if "sampling_params" in params:
        if set(params["sampling_params"]) != SAMPLING_KEYS:
            fail(f"params.sampling_params must have exactly the keys {sorted(SAMPLING_KEYS)}")
    if baseline == "sampling" and not params["sampling_params"]["temperature"] > 0:
        fail("baseline sampling needs sampling_params.temperature > 0")
    if baseline == "verbalized" and params["max_retries"] > 0 and not params["retry_temperature"] > 0:
        fail("retry_temperature must be > 0: a greedy retry repeats the same answer")
    if baseline == "ntp" and "reasoning_effort" in (judge_cfg.get("sampling_params") or {}):
        fail("baseline ntp is undefined for a reasoning judge, whose first token is not an answer")


def run_judge_baseline(
    dataset_config: DatasetConfig,
    judge_key: str,
    params: dict,
    output_dir: Path,
    input_file: Path,
    *,
    seed: int,
    api_base: str,
) -> None:
    """Score every row of an input file and save the results.

    A row that raises ``RowFailure`` is marked failed with empty output. Any other
    exception cancels the remaining rows and is re-raised after ``raw_outputs.json`` is
    saved; ``responses.json`` is not written.

    Args:
        dataset_config: Configuration of the dataset being judged.
        judge_key: Name of a model-config under ``experiments/model-configs/vllm_judge/``.
        params: Validated experiment params (see ``validate_params``).
        output_dir: Directory to write results to.
        input_file: JSON list of extracted measurements.
        seed: The experiment seed.
        api_base: Base URL of the vLLM server, ending in /v1.

    Raises:
        FileExistsError: If ``output_dir`` already holds a ``responses.json``.
        ValueError: If input rows already carry an output field, or a row has no prompt
            because its document or attribute is unknown.
        IndexError: If ``params["row_indices"]`` is out of range for the input.
    """
    judge_cfg = paths.load_model_config("vllm_judge", judge_key)
    model_id = judge_cfg["model_id"]
    baseline = params["baseline"]
    method_params = {k: params[k] for k in sorted(METHOD_PARAMS[baseline])}

    if (output_dir / "responses.json").exists():
        raise FileExistsError(f"{output_dir / 'responses.json'} already exists")
    print(f"Input   : {input_file}")
    with open(input_file) as f:
        all_rows: list[dict] = json.load(f)
    # Rows keep their index in the input file, so a subset run uses the full run's seeds.
    row_ids: list[int] = params.get("row_indices", list(range(len(all_rows))))
    data = [all_rows[i] for i in row_ids]
    clash = {k for row in data for k in OUTPUT_FIELDS if k in row}
    if clash:
        raise ValueError(f"{input_file}: rows already carry output field(s) {sorted(clash)}")

    documents = judge_prompts.load_documents_for_dataset(dataset_config, params["ocr_dir"])
    print(f"Documents: {len(documents)} loaded from {params['ocr_dir']}")
    entries = judge_prompts.prepare_chat_entries(data, documents, dataset_config)
    positions = [int(e["custom_id"]) for e in entries]
    if sorted(positions) != list(range(len(data))):
        missing = [row_ids[p] for p in sorted(set(range(len(data))) - set(positions))]
        raise ValueError(
            f"{len(missing)}/{len(data)} rows have no judge prompt (unknown document or "
            f"attribute), e.g. row indices {missing[:10]}"
        )
    entry_ids = [row_ids[p] for p in positions]

    if baseline == "verbalized":
        entries = to_confidence_entries(entries, dataset_config)
        scorer = score_verbalized
    elif baseline == "sampling":
        scorer = score_sampling
    else:
        scorer = score_ntp
    if baseline == "ntp":
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        label_ids = label_token_ids(tokenizer)
        scorer_kwargs = method_params | {"tokenizer": tokenizer, "label_ids": label_ids}
    else:
        scorer_kwargs = method_params | {
            "base_seed": seed,
            "reasoning_effort": (judge_cfg.get("sampling_params") or {}).get("reasoning_effort"),
        }

    print(f"Scoring {len(entries)} rows with {baseline!r} at {api_base} (model: {model_id}), "
          f"max_concurrent={judge_cfg['max_concurrent']}\n")
    start_time = time.time()

    async def run_all() -> list[asyncio.Task]:
        """Score every entry concurrently, stopping at the first error.

        Returns:
            One finished or cancelled task per entry, in entry order. A task's result is
            a scorer result or a ``RowFailure``.
        """
        sem = asyncio.Semaphore(judge_cfg["max_concurrent"])

        async def one_row(entry: dict, idx: int) -> dict | RowFailure:
            """Score one entry.

            Args:
                entry: Chat entry.
                idx: Row index in the input file.

            Returns:
                The scorer's result, or the failure if its retries are exhausted.
            """
            try:
                return await scorer(client, model_id, entry, idx, sem, **scorer_kwargs)
            except RowFailure as e:
                return e

        async with AsyncOpenAI(
            api_key="EMPTY", base_url=api_base, timeout=judge_cfg["request_timeout"]
        ) as client:
            tasks = [asyncio.create_task(one_row(e, i)) for e, i in zip(entries, entry_ids, strict=True)]
            _, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
        return tasks

    scored: dict[int, dict] = {}
    failed: dict[int, RowFailure] = {}
    errors: dict[int, BaseException] = {}
    n_cancelled = 0
    for i, task in zip(entry_ids, asyncio.run(run_all()), strict=True):
        if task.cancelled():
            n_cancelled += 1
        elif task.exception() is not None:
            errors[i] = task.exception()
        elif isinstance(task.result(), RowFailure):
            failed[i] = task.result()
        else:
            scored[i] = task.result()
    assert len(scored) + len(failed) + len(errors) + n_cancelled == len(data)

    raw: dict[int, dict] = dict(scored)
    raw |= {i: {"failed": True, "error": str(r), "detail": r.detail} for i, r in failed.items()}
    raw |= {i: {"aborted": True, "error": repr(r)} for i, r in errors.items()}
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "raw_outputs.json", "w") as f:
        json.dump({i: raw[i] for i in sorted(raw)}, f, indent=2, ensure_ascii=False)
    if errors:
        idx, error = next(iter(errors.items()))
        print(f"\nABORTED by {len(errors)} error(s), {n_cancelled}/{len(data)} rows not run; first at row {idx}")
        raise error
    assert n_cancelled == 0

    judged: list[dict] = []
    for i, record in zip(row_ids, data, strict=True):
        if i in failed:
            output = dict.fromkeys(OUTPUT_FIELDS) | {"judgement_failed": True}
        else:
            r = scored[i]
            assert 0.0 <= r["score"] <= 1.0, (i, r["score"])
            output = {
                "judgement_p_true": r["score"],
                "judgement_raw_text": r["raw_text"],
                "judgement_model": r["model"],
                "judgement_failed": False,
            }
        judged.append(record | output)
    assert len(judged) == len(data)
    responses_file = output_dir / "responses.json"
    with open(responses_file, "w") as f:
        json.dump(judged, f, indent=4, ensure_ascii=False)
    print(f"Responses saved to {responses_file}")

    print(f"\nFailed rows: {len(failed)}/{len(data)}")
    for i, r in list(sorted(failed.items()))[:10]:
        print(f"    {r}")

    results = list(scored.values())
    summary: dict[str, Any] = {
        "n_rows": len(data),
        "n_failed_rows": len(failed),
        "failed_row_indices": sorted(failed),
        "row_indices": params.get("row_indices"),
    }
    if baseline in ("verbalized", "sampling"):
        summary["n_rejected_answers"] = sum(len(r["rejected"]) for r in results)
        summary["n_rows_with_rejected_answers"] = sum(bool(r["rejected"]) for r in results)
    if baseline == "verbalized":
        # A judge answering on a 0-1 scale shows up as a pile-up at 0 and 1.
        counts = Counter(round(r["score"] * 100) for r in results)
        summary["confidence_counts"] = {str(k): counts[k] for k in sorted(counts)}
    elif baseline == "sampling":
        summary["n_nonliteral_samples"] = sum(r["n_nonliteral"] for r in results)
    else:
        masses = sorted(r["label_mass"] for r in results)
        summary |= {
            "label_token_ids": label_ids,
            "label_mass_min": masses[0] if masses else None,
            "label_mass_median": masses[len(masses) // 2] if masses else None,
            "n_rows_with_label_lookup": sum(bool(r["labels_looked_up"]) for r in results),
            "n_rows_top_token_not_label": sum(not r["top_token_is_label"] for r in results),
        }
    write_run_metadata(
        output_dir,
        start_time=start_time,
        dataset=dataset_config.name,
        extraction_id=params.get("extraction_id"),
        input_file=str(input_file),
        ocr_dir=params["ocr_dir"],
        seed=seed,
        baseline=baseline,
        # Suffixed so that the run is never mistaken for a true/false vote by this judge.
        judge_model=f"{judge_key}-{baseline}",
        judge_key=judge_key,
        judge_model_id=model_id,
        method_params=method_params,
        max_prompt_tokens=max((r["prompt_tokens"] for r in results), default=None),
        **summary,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        Parser with a positional ``config`` path and an optional ``--api-base`` URL.
    """
    p = argparse.ArgumentParser(
        description="Score extracted measurements with a baseline confidence method.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("config", help="Path to an experiment-configs/.../<id>.yaml.")
    p.add_argument(
        "--api-base", default=None, metavar="URL",
        help="Base URL of the vLLM server; takes precedence over params.api_base.",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    """Run the experiment described by a config file.

    Args:
        argv: Command-line arguments; None reads ``sys.argv``.

    Raises:
        ValueError: If the config is invalid, its seed is not the repo-wide seed, its
            dataset does not match its extraction run, or no server URL is given.
        FileNotFoundError: If the input file or a referenced config does not exist.
    """
    args = build_parser().parse_args(argv)
    config_path = Path(args.config)
    cfg = paths.load_experiment_config(config_path)
    params, seed = cfg["params"], cfg["seed"]
    paths.require_params(params, "judge", config_path=config_path)
    validate_params(params, paths.load_model_config("vllm_judge", params["judge"]), config_path)

    repo_seed = paths.load_config()["defaults"]["seed"]
    if seed != repo_seed:
        raise ValueError(f"{config_path}: seed {seed} does not match the repo-wide seed {repo_seed}")
    set_seeds(seed)

    api_base = args.api_base or params.get("api_base")
    if api_base is None:
        raise ValueError(f"{config_path}: no server URL; pass --api-base or set params.api_base")

    dataset = params["dataset"]
    dataset_config = load_dataset_config(dataset)
    if "extraction_id" in params:
        extraction_dir = paths.find_result_dir(params["extraction_id"])
        if extraction_dir.parts[-3] != dataset:
            raise ValueError(
                f"{config_path}: params.dataset {dataset!r} does not match the dataset of "
                f"params.extraction_id ({extraction_dir.parts[-3]!r})"
            )
        input_file = extraction_dir / "final.json"
    else:
        input_file = Path(params["synthetic_file"])
    if not input_file.exists():
        raise FileNotFoundError(f"input file not found: {input_file}")

    output_dir = paths.result_dir(dataset, "judge_baselines", cfg["id"])
    print(f"\nDataset          : {dataset}")
    print(f"Baseline         : {params['baseline']}")
    print(f"Judge            : {params['judge']}")
    print(f"API base         : {api_base}")
    print(f"Output           : {output_dir}\n")
    run_judge_baseline(
        dataset_config=dataset_config,
        judge_key=params["judge"],
        params=params,
        output_dir=output_dir,
        input_file=input_file,
        seed=seed,
        api_base=api_base,
    )


if __name__ == "__main__":
    main()
