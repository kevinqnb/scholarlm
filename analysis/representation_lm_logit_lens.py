"""Difference-in-means + logit lens on RepresentationLM key-term representations.

For a single collected layer (``--layer``), compute the mean representation of
each key term (pond / lake / wetland), form the pairwise difference vectors
(pond-lake, pond-wetland, lake-wetland), and read each difference direction
through the model's unembedding — a logit lens — reporting the top-k tokens it
promotes.

Method
------
Each difference vector ``d = mu_a - mu_b`` is a direction in the residual
stream at layer ``L`` (pre-final-norm for L < n_layers). The model turns a
final residual into logits as ``(RMSNorm(h) @ W_U^T)`` where
``RMSNorm(x) = (x * rsqrt(mean(x^2) + eps)) * g``, ``g = model.norm.weight``.
We report TWO projections of each ``d``:

  * **with final-norm gain**:  ``(d * g) @ W_U^T``  — the faithful read-out up
    to a positive scalar (the ``rsqrt(...)`` factor is a single positive number
    for a fixed direction, so it changes neither the token ranking nor the
    relative logits within one projection; it is omitted, not approximated).
  * **without gain**:  ``d @ W_U^T``  — the raw logit lens.

If the two disagree on the top-k, that is a result, not a footnote.

For the POST-final-norm layer (``L == n_layers``) the representation already
went through RMSNorm at collection time, so the faithful logit lens is the
plain ``d @ W_U^T`` and the gain must NOT be re-applied — the script switches
to a single "raw" projection there.

These logits are DIFFERENCES. A "top token" for ``pond - lake`` is one the
mean-pond representation promotes *relative to* mean-lake — not a token the
model would actually predict next.

Findings (2026-09-02, pond / llama-3.1-8b-base / 2026_09_01 artifact)
  * layer 32 (post-final-norm): clean known-answer PASS. +pond →
    ` pond`/` ponds`/` created`/` built`/` constructed`; +lake → ` Tahoe`/
    ` Erie`/` Superior`/` Michigan`/` Ontario`/` Huron`. This is the layer to
    use for a logit-lens read.
  * layers 8, 16, 24 (pre-final-norm): the difference-in-means signal is real
    and enormous (``||d||`` sits 200-310 sigma above the shuffled null) but it
    does NOT project onto interpretable tokens through the frozen unembedding —
    the top-k is incoherent subword fragments, gain or no gain. This is the
    expected failure of the raw logit lens off the final layer (cf. the tuned
    lens / this repo's JacobianLensLM). A mid-layer read needs one of those,
    not this script.

Known-answer prediction (state before running; a miss means STOP and debug the
projection, do not write up whatever appeared):
  * pond - lake, with-gain, +d top-10 should contain ` pond` and/or ` ponds`;
    -d top-10 should contain ` lake` and/or ` Lake`.
  * wetland's surface form is NOT a single token (`wet`+`land`), so do not
    expect ` wetland` in the wetland pairs — `land`/`lands`/marsh/swamp-type
    tokens are the tell there.
  * Layer 8 is only a quarter of the way up the stack; a weak or null result
    at this layer is a plausible true outcome, not necessarily a bug.

Controls
--------
  * cos(mu_a, mu_b) and the mean norms are printed FIRST. If the class means
    are nearly parallel (cos ~ 0.99), ``d`` is a small residual of a large
    cancellation and the token list must be read with that caveat.
  * Shuffled-label null: key-term labels are permuted across all n rows (class
    sizes preserved), the class means and difference vectors recomputed, over
    many seeds. ``||d||`` for the real labels should sit far outside the
    shuffled distribution; one shuffled draw is also logit-lensed to show its
    top tokens are incoherent.

Usage
-----
    uv run python analysis/representation_lm_logit_lens.py \
        --npz data/experiments/pond/representation_lm/llama-3.1-8b-base/<date>/representations.npz \
        --layer 8 [--topk 10] [--n-shuffle 20]

Writes ``representation_lm_logit_lens_layer{L:02d}.{txt,csv}`` to
``data/experiments/{dataset}/analysis/``. The CSV
(pair, projection, direction, rank, token_id, token_repr, decoded, logit) is
the durable, re-checkable artifact.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))
import paths


def _load_unembedding_and_norm(
    model_name: str, cache_dir: str | None
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Read ``lm_head.weight`` (full), ``model.norm.weight``, and rms_norm_eps /
    vocab_size straight from the cached HF checkpoint — never via a live model
    (meta-tensor / offload hazards; see jacobianlenslm.load_unembedding_row)."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError
    from safetensors import safe_open
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
    vocab_size = int(cfg.vocab_size)
    eps = float(cfg.rms_norm_eps)

    def _shard_for(key: str) -> str:
        try:
            idx = hf_hub_download(model_name, "model.safetensors.index.json", cache_dir=cache_dir)
            wmap = json.load(open(idx))["weight_map"]
            if key not in wmap:
                raise KeyError(
                    f"{model_name} checkpoint has no {key!r} "
                    f"(found {sorted(k for k in wmap if 'lm_head' in k or 'norm' in k or 'embed' in k)!r})"
                )
            return hf_hub_download(model_name, wmap[key], cache_dir=cache_dir)
        except EntryNotFoundError:
            return hf_hub_download(model_name, "model.safetensors", cache_dir=cache_dir)

    lm_shard = _shard_for("lm_head.weight")
    norm_shard = _shard_for("model.norm.weight")
    with safe_open(lm_shard, framework="pt", device="cpu") as f:
        W_U = f.get_slice("lm_head.weight")[:, :].float().numpy()
    with safe_open(norm_shard, framework="pt", device="cpu") as f:
        g = f.get_slice("model.norm.weight")[:].float().numpy()
    return W_U, g, eps, vocab_size


def _top_tokens(logits: np.ndarray, tokenizer, k: int) -> list[tuple[int, str, str, float]]:
    idx = np.argpartition(-logits, range(k))[:k]
    idx = idx[np.argsort(-logits[idx])]
    out = []
    for i in idx:
        i = int(i)
        out.append((i, tokenizer.convert_ids_to_tokens(i), tokenizer.decode([i]), float(logits[i])))
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--npz", required=True, type=Path, help="Path to representations.npz")
    p.add_argument("--layer", required=True, type=int, help="Which collected layer to analyse (e.g. 8).")
    p.add_argument("--topk", type=int, default=10, help="Tokens to report per direction (default 10).")
    p.add_argument("--n-shuffle", type=int, default=20, help="Shuffled-label null draws (default 20).")
    p.add_argument("--seed", type=int, default=342, help="Base RNG seed for the shuffled null.")
    args = p.parse_args(argv)

    import os
    from transformers import AutoTokenizer

    hf_cache = os.environ.get("HF_CACHE")

    z = np.load(args.npz, allow_pickle=False)
    available = [int(L) for L in z["layers"]]
    if args.layer not in available:
        raise SystemExit(f"layer {args.layer} not in this npz; available: {available}")

    labels = z["labels"]
    terms = [str(t) for t in z["key_terms"]]
    model_name = z["model_name"].item()
    hidden = int(z["hidden_size"])
    n = len(labels)
    assert set(np.unique(labels)) <= set(terms)
    assert len(terms) == 3, f"this script assumes exactly 3 key terms, got {terms}"

    # Per-layer sanity: cross-check mean row L2 against run_metadata (catches a
    # mislabelled npz member). Missing metadata is a hard error.
    meta_path = args.npz.parent / "run_metadata.json"
    if not meta_path.exists():
        raise SystemExit(f"missing {meta_path}")
    meta = json.loads(meta_path.read_text())
    X = z[f"rep_layer_{args.layer:02d}"].astype(np.float64)
    assert X.shape == (n, hidden), X.shape
    assert np.isfinite(X).all()
    obs_l2 = float(np.linalg.norm(X, axis=1).mean())
    exp_l2 = float(meta["mean_row_l2_by_layer"][str(args.layer)])
    assert abs(obs_l2 - exp_l2) <= 1e-3 * exp_l2, (obs_l2, exp_l2)

    counts = {t: int((labels == t).sum()) for t in terms}
    assert counts == meta["rows_per_term"], (counts, meta["rows_per_term"])

    W_U, g, eps, vocab_size = _load_unembedding_and_norm(model_name, hf_cache)
    assert W_U.shape == (vocab_size, hidden), (W_U.shape, vocab_size, hidden)
    assert g.shape == (hidden,), g.shape
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=hf_cache)

    def class_means(lab: np.ndarray) -> dict[str, np.ndarray]:
        return {t: X[lab == t].mean(axis=0) for t in terms}

    mu = class_means(labels)
    pairs = [(terms[0], terms[1]), (terms[0], terms[2]), (terms[1], terms[2])]

    is_post_norm = args.layer == available[-1]

    def project(d: np.ndarray, variant: str) -> np.ndarray:
        # "gain": (d * model.norm.weight) @ W_U^T  — approximates the final-norm
        #         read-out for a PRE-norm residual (rank-invariant rsqrt omitted).
        # "raw":  d @ W_U^T. For a POST-norm layer this IS the faithful logit
        #         lens (the residual already went through RMSNorm at collection).
        assert variant in ("gain", "raw")
        v = d * g if variant == "gain" else d
        return v @ W_U.T

    # primary projection: 'raw' for the post-norm layer, 'gain' otherwise.
    primary = "raw" if is_post_norm else "gain"
    variants = ["raw"] if is_post_norm else ["gain", "raw"]
    VTAG = {
        "gain": "with final-norm gain:  (d * model.norm.weight) @ W_U^T",
        "raw": "raw:  d @ W_U^T" + ("  (post-norm layer — faithful logit lens)" if is_post_norm else ""),
    }

    # ---- shuffled-label null ------------------------------------------------
    real_dnorm = {pr: float(np.linalg.norm(mu[pr[0]] - mu[pr[1]])) for pr in pairs}
    shuf_dnorms: dict[tuple, list[float]] = {pr: [] for pr in pairs}
    first_shuf_proj: dict[tuple, np.ndarray] = {}
    for s in range(args.n_shuffle):
        rng = np.random.default_rng(args.seed + s)
        lab_s = rng.permutation(labels)
        mu_s = class_means(lab_s)
        for pr in pairs:
            d_s = mu_s[pr[0]] - mu_s[pr[1]]
            shuf_dnorms[pr].append(float(np.linalg.norm(d_s)))
            if s == 0:
                first_shuf_proj[pr] = project(d_s, primary)

    # ---- report ----------------------------------------------------------------
    out_dir = paths.analysis_dir(_dataset_from_npz(args.npz))
    out_dir.mkdir(parents=True, exist_ok=True)
    txt_path = out_dir / f"representation_lm_logit_lens_layer{args.layer:02d}.txt"
    csv_path = out_dir / f"representation_lm_logit_lens_layer{args.layer:02d}.csv"

    lines: list[str] = []
    def emit(s: str = "") -> None:
        lines.append(s)
        print(s)

    emit(f"RepresentationLM logit lens — difference-in-means, layer {args.layer}")
    emit(f"model: {model_name}")
    emit(f"n={n}  ({', '.join(f'{t} {counts[t]}' for t in terms)})")
    emit(f"npz: {args.npz}")
    emit(f"layer {args.layer} is {'POST-final-norm' if is_post_norm else 'PRE-final-norm'} "
         f"(mean row L2 {obs_l2:.4g})")
    emit("")
    emit("logits are DIFFERENCES: a top token for 'a - b' is promoted by mean-a")
    emit("RELATIVE TO mean-b, not an absolute next-token prediction.")
    for v in variants:
        emit(f"  {VTAG[v]}")
    emit(f"  primary projection: {primary}")
    emit("")

    emit("--- class-mean geometry ---")
    for t in terms:
        emit(f"  ||mu_{t}||           = {np.linalg.norm(mu[t]):.4g}")
    for a, b in pairs:
        c = float(mu[a] @ mu[b] / (np.linalg.norm(mu[a]) * np.linalg.norm(mu[b])))
        emit(f"  cos(mu_{a}, mu_{b}) = {c:+.4f}    ||mu_{a} - mu_{b}|| = {real_dnorm[(a, b)]:.4g}")
    emit("")

    emit(f"--- shuffled-label null ({args.n_shuffle} permutations of all {n} labels, sizes preserved) ---")
    emit(f"  {'pair':<20}{'||d_real||':>12}{'||d_shuf|| mean':>16}{'std':>10}{'max':>10}{'(real-mean)/std':>17}")
    for pr in pairs:
        arr = np.array(shuf_dnorms[pr])
        z_ = (real_dnorm[pr] - arr.mean()) / arr.std()
        emit(f"  {pr[0]+' - '+pr[1]:<20}{real_dnorm[pr]:>12.4g}{arr.mean():>16.4g}"
             f"{arr.std():>10.3g}{arr.max():>10.4g}{z_:>17.1f}")
    emit("")

    csv_rows = []
    for a, b in pairs:
        d = mu[a] - mu[b]
        emit(f"================  {a} - {b}  ================")
        for variant in variants:
            tag = VTAG[variant]
            logit = project(d, variant)
            emit(f"  [{tag}]")
            for sign, toward in ((+1.0, a), (-1.0, b)):
                emit(f"    {'+d' if sign > 0 else '-d'} — promoted toward {toward}:")
                for rank, (tid, trepr, dec, lg) in enumerate(_top_tokens(sign * logit, tokenizer, args.topk), 1):
                    emit(f"      {rank:>2}. {trepr!r:<18} {dec!r:<16} {lg:+.3f}")
                    csv_rows.append({
                        "pair": f"{a}-{b}", "projection": tag,
                        "direction": f"toward_{toward}",
                        "rank": rank, "token_id": tid, "token_repr": trepr,
                        "decoded": dec, "logit": lg,
                    })
        # shuffled control for this pair (primary projection, first seed)
        emit(f"  [shuffled null, seed {args.seed}, {primary} — expect incoherence]")
        for rank, (tid, trepr, dec, lg) in enumerate(_top_tokens(first_shuf_proj[(a, b)], tokenizer, 5), 1):
            emit(f"      {rank:>2}. {trepr!r:<18} {dec!r:<16} {lg:+.3f}")
        emit("")

    # ---- known-answer check (pond-lake, primary projection) -----------------
    emit(f"--- KNOWN-ANSWER CHECK: pond - lake, {primary} projection ---")
    if {"pond", "lake"} <= set(terms):
        d_pl = mu["pond"] - mu["lake"]
        lg = project(d_pl, primary)
        pos = {t[2].strip().lower() for t in _top_tokens(lg, tokenizer, args.topk)}
        neg = {t[2].strip().lower() for t in _top_tokens(-lg, tokenizer, args.topk)}
        ok_pos = bool(pos & {"pond", "ponds"})
        ok_neg = bool(neg & {"lake", "lakes"})
        emit(f"  +d top-{args.topk} contains pond/ponds : {ok_pos}   ({sorted(pos)})")
        emit(f"  -d top-{args.topk} contains lake/lakes : {ok_neg}   ({sorted(neg)})")
        if ok_pos and ok_neg:
            emit("  => PASS. Projection surfaces the terms themselves; proceed.")
        else:
            emit("  => MISS. Per the docstring: STOP and debug the projection / read point")
            emit("     before interpreting the token lists above. (Or conclude layer "
                 f"{args.layer} genuinely does not localise this — decide, don't cherry-pick.)")
    emit("")

    txt_path.write_text("\n".join(lines) + "\n")
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)
    print(f"\nwrote {txt_path}")
    print(f"wrote {csv_path}")


def _dataset_from_npz(npz: Path) -> str:
    # data/experiments/{dataset}/representation_lm/{model}/{date}/representations.npz
    parts = npz.resolve().parts
    i = parts.index("experiments")
    return parts[i + 1]


if __name__ == "__main__":
    main()
