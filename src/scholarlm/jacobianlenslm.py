import gc
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

try:
    import torch
    from nnterp import StandardizedTransformer
    HAS_GPU_DEPS = True
except ImportError:
    print("Warning: PyTorch and/or nnterp not available; JacobianLensLM will not use GPU acceleration.")
    HAS_GPU_DEPS = False


def load_jacobian_lens(path: str, cache_dir: str | None = None) -> dict[str, Any]:
    """
    Resolve and load a pretrained Jacobian-lens checkpoint.

    ``path`` is either a local filesystem path to a ``.pt`` file, or a
    HuggingFace Hub spec of the form ``repo_id:filename`` (e.g.
    ``neuronpedia/jacobian-lens:llama3.1-8b/jlens/Salesforce-wikitext/Llama-3.1-8B_jacobian_lens.pt``).
    No specific repo or filename is hardcoded here — the caller supplies the
    full spec, so any future lens artifact works without code changes.

    Args:
        path: Local file path, or ``repo_id:filename`` HuggingFace Hub spec.
        cache_dir: Cache directory for Hub downloads (e.g. ``$HF_CACHE``).
            Ignored for local paths.

    Returns:
        The raw checkpoint dict as saved by ``JacobianLens.save()`` in
        Anthropic's reference implementation:
        ``{"J": {layer_idx: Tensor[d_model, d_model]}, "n_prompts": int,
        "source_layers": list[int], "d_model": int}``.

    Raises:
        ValueError: If the resolved file doesn't look like a Jacobian-lens
            checkpoint (missing the ``"J"`` key), or if ``path`` is neither an
            existing local file nor a ``repo_id:filename`` spec.
    """
    local_path = Path(path)
    if local_path.is_file():
        resolved = local_path
    else:
        if ":" not in path:
            raise ValueError(
                f"'{path}' is not an existing local file and is not of the form "
                "'repo_id:filename' for a HuggingFace Hub download."
            )
        repo_id, filename = path.split(":", 1)
        from huggingface_hub import hf_hub_download
        resolved = Path(hf_hub_download(repo_id, filename, cache_dir=cache_dir))

    checkpoint = torch.load(resolved, map_location="cpu", weights_only=True)
    if "J" not in checkpoint:
        raise ValueError(
            f"{resolved} does not look like a Jacobian-lens checkpoint "
            f"(expected key 'J', found {sorted(checkpoint.keys())!r})."
        )
    return checkpoint


def load_unembedding_row(model_name: str, token_id: int, cache_dir: str | None = None) -> "torch.Tensor":
    """
    Read a single row of ``lm_head.weight`` straight from the HF Hub checkpoint,
    bypassing the live (possibly meta-dispatched or CPU-offloaded) model entirely.

    ``StandardizedTransformer(..., device_map="auto")`` only fully materializes a
    module's parameters during that module's own forward call (via accelerate's
    ``AlignDevicesHook``); a bare ``model.lm_head.weight`` attribute read never
    triggers that hook and raises "Cannot copy out of meta tensor; no data!" —
    reliably when the model doesn't fit the GPU and gets CPU-offloaded, and
    silently-by-luck otherwise. ``W_U[true_id, :]`` doesn't depend on any trace
    input, so there's no reason to route it through nnsight at all: fetch the one
    shard that holds ``lm_head.weight`` and slice the row directly.

    Args:
        model_name: HuggingFace repo id (e.g. ``meta-llama/Llama-3.1-8B``).
        token_id: Row index into the unembedding matrix.
        cache_dir: Cache directory for Hub downloads (e.g. ``$HF_CACHE``).

    Returns:
        A 1-D float32 CPU tensor of shape ``[d_model]``.

    Raises:
        KeyError: If ``lm_head.weight`` isn't a key in the checkpoint's weight
            map (e.g. a model with tied embeddings, where the unembedding lives
            at ``model.embed_tokens.weight`` instead).
    """
    import json
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError
    from safetensors import safe_open

    try:
        index_path = hf_hub_download(model_name, "model.safetensors.index.json", cache_dir=cache_dir)
        weight_map = json.load(open(index_path))["weight_map"]
        if "lm_head.weight" not in weight_map:
            raise KeyError(
                f"{model_name}'s checkpoint has no 'lm_head.weight' entry (found "
                f"{sorted(k for k in weight_map if 'embed' in k or 'lm_head' in k)!r}); "
                "the model likely ties the unembedding to the input embeddings."
            )
        shard_file = weight_map["lm_head.weight"]
        shard_path = hf_hub_download(model_name, shard_file, cache_dir=cache_dir)
    except EntryNotFoundError:
        # Single-shard checkpoints have no index file at all.
        shard_path = hf_hub_download(model_name, "model.safetensors", cache_dir=cache_dir)

    with safe_open(shard_path, framework="pt", device="cpu") as f:
        row = f.get_slice("lm_head.weight")[token_id, :]
    return row.float()


def tokenize(
    instructions: str,
    context: str,
    query: str,
    tokenizer: callable,
    use_chat_template: bool = False,
) -> tuple[list[int], list[int], list[int], list[int]]:
    """
    Apply an (instructions, query, context) triple to a prompt template and return
    the tokenized input along with the token indices for each section.

    Unlike ``judgementlm.tokenize()``, the context comes *last*: the Jacobian
    lens is meant to probe how the model processes context tokens once it has
    already seen the extraction being validated (see thread note point 3 under
    Thrust 1), so the query must precede the context here.

    Args:
        instructions (str): The instruction string.
        context (str): The context string.
        query (str): The query string.
        tokenizer (Callable): HuggingFace tokenizer.
        use_chat_template (bool): If False (the default here — base models don't
            reliably follow an inherited chat template), tokenize the raw content
            string directly, with an explicit BOS token added via
            ``add_special_tokens=True``. The pretrained Jacobian lens was fit on
            raw-text prompts with BOS included (Anthropic's reference
            implementation sets ``tokenizer.add_bos_token = True`` for exactly
            this reason), so omitting it would score off-distribution. If True,
            mirrors ``judgementlm.tokenize()``'s convention: the chat template's
            formatted string already contains literal control tokens, so
            ``add_special_tokens=False`` avoids double-adding them.

    Returns:
        (tokenized_chat, instruction_tokens, context_tokens, query_tokens)
    """
    content = f"## INSTRUCTIONS:\n{instructions}\n\n## QUERY:\n{query}\n\n## CONTEXT:\n{context}"
    if use_chat_template:
        chat = [{"role": "user", "content": content}]
        formatted_chat = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
        add_special_tokens = False
    else:
        formatted_chat = content
        add_special_tokens = True
    tokenized_chat = tokenizer(
        formatted_chat, return_offsets_mapping=True, add_special_tokens=add_special_tokens
    )

    instruction_start = formatted_chat.index("## INSTRUCTIONS:\n") + len("## INSTRUCTIONS:\n")
    instruction_end = formatted_chat.index("\n\n## QUERY:")
    query_start = formatted_chat.index("## QUERY:\n") + len("## QUERY:\n")
    query_end = formatted_chat.index("\n\n## CONTEXT:")
    context_start = formatted_chat.index("## CONTEXT:\n") + len("## CONTEXT:\n")
    context_end = context_start + len(context)

    instruction_tokens = [
        i for i, (s, e) in enumerate(tokenized_chat["offset_mapping"])
        if s >= instruction_start and e <= instruction_end
    ]
    context_tokens = [
        i for i, (s, e) in enumerate(tokenized_chat["offset_mapping"])
        if s >= context_start and e <= context_end
    ]
    query_tokens = [
        i for i, (s, e) in enumerate(tokenized_chat["offset_mapping"])
        if s >= query_start and e <= query_end
    ]

    return tokenized_chat["input_ids"], instruction_tokens, context_tokens, query_tokens


class JacobianLensLM:
    """
    Computes and persists Jacobian-lens j-scores for individual examples.

    Given a pretrained per-layer Jacobian estimate Z_l for a model, this class
    scores each context token t at each fitted layer l as
    ``S[l, t] = <W_U[true_id, :], norm(Z_l @ h_{l,t})>``, where ``h_{l,t}`` is
    the residual-stream output at layer l, position t, and ``norm`` is the
    model's own final-layer normalization. Applying the model's own norm to
    the transported vector *before* projecting onto the 'true' direction
    (rather than a raw ``<W_U[true_id,:] @ Z_l, h_{l,t}>`` dot product) is
    what makes scores comparable across layers, which otherwise have no
    shared scale — see the "FIX" note under Thrust 1 in
    ``notes/scholarlm/threads/Jacobian Lens.md``. The FIX note's formula also
    calls for a full-vocab softmax on top of this; that's deliberately
    dropped here — the softmax denominator only corrects a much smaller
    residual scale effect on top of what ``norm`` already fixes, at the cost
    of an extra ``[n_context, d_model] @ [d_model, n_vocab]`` matmul per
    layer (~31x the cost of the ``Z_l @ h`` transport itself, and a
    transient buffer that scales with vocab size — tens of GB per layer at
    this repo's longer document lengths).

    This is deliberately not a subclass or extension of ``JudgementLM``: the
    prompt ordering (query before context, vs. ``JudgementLM.generate()``'s
    context before query), the trace shape (single prefill pass over every
    context position, vs. per-generated-token capture), and the true-token
    lookup (single lowercase id, vs. four case variants) all diverge enough
    that sharing the class would have meant threading toggles through code
    neither use case actually shares.

    Args:
        model_name (str): The name of the model to load from NNsight or huggingface.
        jacobian_lens_path (str): Local path or ``repo_id:filename`` HuggingFace
            Hub spec for the pretrained Jacobian-lens checkpoint (see
            ``load_jacobian_lens``).
        nnsight_kwargs (dict): Additional keyword arguments to pass to the NNsight LanguageModel.
        use_chat_template (bool): Whether to wrap prompts with the tokenizer's chat
            template. Default False. See ``tokenize()``.
        hf_cache_dir (str | None): Cache directory for Hub lens downloads.
        verbose (bool): Whether to print verbose output. Default False.
    """
    def __init__(
        self,
        model_name: str,
        jacobian_lens_path: str,
        nnsight_kwargs: dict = {},
        use_chat_template: bool = False,
        hf_cache_dir: str | None = None,
        verbose: bool = False,
    ):
        self.model_name = model_name
        self.verbose = verbose
        self.use_chat_template = use_chat_template

        self._setup_devices()

        self.llm = StandardizedTransformer(model_name, enable_attention_probs=False, **nnsight_kwargs)
        print(self.llm)
        self.tokenizer = self.llm.tokenizer
        if getattr(self.tokenizer, "add_bos_token", None) is not None:
            self.tokenizer.add_bos_token = True
        self.n_layers = len(self.llm.model.layers)
        self.hidden_size = self.llm.config.hidden_size

        self.max_prompt_tokens: int = 0

        true_ids = self.tokenizer.encode("true", add_special_tokens=False)
        if len(true_ids) != 1:
            print(f"Warning: 'true' tokenizes to {len(true_ids)} tokens {true_ids}; using first token {true_ids[0]}.")
        self.true_token_id = true_ids[0]

        checkpoint = load_jacobian_lens(jacobian_lens_path, cache_dir=hf_cache_dir)
        self._validate_lens(checkpoint)
        self.source_layers: list[int] = sorted(checkpoint["J"].keys())

        self.jacobian_matrices: dict[int, torch.Tensor] = self._load_jacobian_matrices(checkpoint)

        # W_U[true_id, :] doesn't depend on any trace input, so it's read
        # straight from the checkpoint shard rather than off the live nnsight
        # model — see load_unembedding_row()'s docstring for why the obvious
        # `self.llm.lm_head.weight[...]` read is unreliable under
        # device_map="auto" (breaks whenever accelerate CPU-offloads part of
        # the model to fit a small GPU).
        w_u_true = load_unembedding_row(model_name, self.true_token_id, cache_dir=hf_cache_dir)
        self._w_u_true = w_u_true.to(self.tensor_device)


    def _setup_devices(self):
        """
        Set up device allocation for LLM and tensors.
        If multiple GPUs are available, use separate devices for LLM and tensors.
        Otherwise, use the same device for both.
        """
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            if (num_gpus >= 2):
                self.llm_device = torch.device("cuda:0")
                self.tensor_device = torch.device("cuda:1")
                if self.verbose:
                    print(f"Using {num_gpus} GPUs: LLM on cuda:0, tensors on cuda:1")
            else:
                self.llm_device = torch.device("cuda:0")
                self.tensor_device = torch.device("cuda:0")
                if self.verbose:
                    print(f"Using single GPU: cuda:0")
        else:
            self.llm_device = torch.device("cpu")
            self.tensor_device = torch.device("cpu")
            if self.verbose:
                print("No GPU available, using CPU")


    def _validate_lens(self, checkpoint: dict[str, Any]) -> None:
        """
        Validate a loaded Jacobian-lens checkpoint against this instance's model.

        Raises a clear error immediately rather than letting a shape mismatch
        surface as an opaque failure deep inside a later matmul.

        Args:
            checkpoint: The raw checkpoint dict from ``load_jacobian_lens``.

        Raises:
            ValueError: If ``d_model`` doesn't match this model's hidden size,
                if a layer index is out of range for this model, or if a
                layer's Jacobian isn't a square ``[d_model, d_model]`` matrix.
        """
        d_model = checkpoint["d_model"]
        if d_model != self.hidden_size:
            raise ValueError(
                f"Jacobian lens d_model={d_model} does not match "
                f"{self.model_name}'s hidden_size={self.hidden_size}."
            )
        for layer_idx, J in checkpoint["J"].items():
            if not (0 <= layer_idx < self.n_layers):
                raise ValueError(
                    f"Jacobian lens layer index {layer_idx} out of range for a "
                    f"{self.n_layers}-layer model."
                )
            if tuple(J.shape) != (d_model, d_model):
                raise ValueError(
                    f"Jacobian lens layer {layer_idx} has shape {tuple(J.shape)}, "
                    f"expected ({d_model}, {d_model})."
                )


    def _load_jacobian_matrices(self, checkpoint: dict[str, Any]) -> dict[int, torch.Tensor]:
        """
        Load each fitted layer's Jacobian estimate Z_l onto ``self.tensor_device``.

        Cast fp16 (as saved in the checkpoint) to fp32 up front: ``compute_scores``
        needs the full ``Z_l @ h_{l,t}`` vector (not just a single projected
        scalar), so unlike the previous single-row-projection design there's no
        way to avoid holding every fitted layer's full ``[d_model, d_model]``
        matrix resident for the lifetime of this instance.

        Args:
            checkpoint: The raw checkpoint dict from ``load_jacobian_lens``.

        Returns:
            Dict mapping layer index to a ``[d_model, d_model]`` float32
            tensor on ``self.tensor_device``.
        """
        return {
            layer_idx: checkpoint["J"][layer_idx].float().to(self.tensor_device)
            for layer_idx in self.source_layers
        }


    def compute_scores(
        self,
        instructions: str,
        context: str,
        query: str,
    ) -> dict[str, Any]:
        """
        Compute the j-score matrix S for a single (instructions, context, query)
        triple, restricted to context-token positions.

        Runs a prefill-only NNsight trace (no generation — S only needs
        per-position residuals from the forward pass). For each source layer,
        transports that layer's context-token residuals through Z_l, routes
        the result through the model's own final norm, and projects onto the
        fixed 'true' unembedding direction — so the full d_model-width
        residual is never retained outside a single loop iteration (see the
        memory-constraint note in the build's design notes: at this repo's
        real context lengths, storing it across all layers would be
        multi-terabyte territory across a full dataset).

        Args:
            instructions (str): The instructions string.
            context (str): The context string.
            query (str): The query string.

        Returns:
            dict: A dictionary containing:
                'S' (np.ndarray): j-score matrix, shape
                    ``(len(source_layers), n_context_tokens)``.
                'layer_indices' (np.ndarray): the source-layer index each row
                    of ``S`` corresponds to (``S`` is not indexed by
                    ``range(n_layers)`` — the lens does not necessarily cover
                    every layer of the model).
        """
        (tokenized_prompt,
         instruction_token_indices,
         context_token_indices,
         query_token_indices) = tokenize(
            instructions, context, query, self.tokenizer, use_chat_template=self.use_chat_template
        )
        prompt_len = len(tokenized_prompt)
        self.max_prompt_tokens = max(self.max_prompt_tokens, prompt_len)
        n_context = len(context_token_indices)
        tensor_device = self.tensor_device

        # torch.no_grad() wraps the *entire* trace, not just the code inside
        # this `with` block: nnsight defers the actual model forward pass to
        # the trace's __exit__ (its "backend" execution), so a no_grad only
        # around the statements written inside the block does not cover that
        # deferred forward computation. Without this, the base model's own
        # 32-layer forward pass builds a full autograd graph (retaining
        # per-layer attention/MLP activations for a backward() this class
        # never calls) — roughly doubling peak memory and, on this repo's
        # longer documents, OOMing even after the logits_to_keep fix below
        # removed the other major waste.
        #
        # logits_to_keep=1: this trace only ever reads per-layer residuals
        # (layer.output[0]) — .output.logits is never touched. HF's default
        # (logits_to_keep=0) is a footgun that means "compute logits for
        # every position" (`slice(-0, None) == slice(0, None)`, not "skip
        # this"), so without this the model's own forward pass wastes a
        # `[seq_len, n_vocab]` unembed on the *entire* prompt every trace.
        with torch.no_grad(), self.llm.trace(tokenized_prompt, logits_to_keep=1) as tracer:
            scores = torch.zeros(
                size=(len(self.source_layers), n_context),
                device=tensor_device,
                dtype=torch.float32,
            ).save()

            for row, layer_idx in enumerate(self.source_layers):
                layer = self.llm.model.layers[layer_idx]
                layer_out = layer.output[0]
                if layer_out.ndim == 3:  # (batch, seq_len, hidden_size)
                    ctx_out = layer_out[0, context_token_indices, :]
                else:  # (seq_len, hidden_size)
                    ctx_out = layer_out[context_token_indices, :]
                ctx_out = ctx_out.detach()

                # z_{l,t} = Z_l @ h_{l,t}: transport the residual through
                # this layer's Jacobian estimate. Done in fp32 on
                # tensor_device to match self.jacobian_matrices' precision,
                # then cast back to the model's own dtype/device before
                # feeding it through the model's real ln_final module below.
                z = ctx_out.float().to(tensor_device) @ self.jacobian_matrices[layer_idx].T
                z = z.to(device=self.llm_device, dtype=ctx_out.dtype)

                # norm(z), via the model's own final-layer-norm module (not a
                # freestanding RMSNorm reimplementation) so this stays
                # correct under device_map="auto" CPU offloading: calling a
                # module through its normal forward goes through
                # accelerate's AlignDevicesHook, unlike a bare `.weight`
                # attribute read.
                ln_out = self.llm.ln_final(z)

                # <W_U[true_id,:], norm(z)> — a fixed precomputed direction,
                # not the full softmax(W_U @ norm(z)); see the class
                # docstring for why that's dropped.
                scores[row, :] = (ln_out.float().detach().to(tensor_device) @ self._w_u_true).detach()

        S = scores.detach().float().cpu().numpy()

        del tracer
        del scores
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return {
            "S": S,
            "layer_indices": np.array(self.source_layers, dtype=np.int64),
        }


    def predict(
        self,
        prompts: list[tuple[str, str, str]],
    ) -> list[dict[str, Any]]:
        """
        Run ``compute_scores()`` on a batch of (instructions, context, query) triples.

        Args:
            prompts (list[tuple[str, str, str]]): A list of (instructions, context, query) triples.

        Returns:
            list[dict]: One score dict per prompt. See ``compute_scores()`` for the dict structure.
        """
        results = []
        for instructions, context, query in tqdm(prompts):
            results.append(self.compute_scores(instructions, context, query))
        return results
