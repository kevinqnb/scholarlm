import gc

import numpy as np
import torch
import torch.nn.functional as F
from nnterp import StandardizedTransformer
from tqdm import tqdm


def tokenize(
    instructions: str,
    context: str,
    query: str,
    tokenizer: callable,
) -> tuple[list[int], list[int], list[int], list[int]]:
    """
    Apply a chat template to an (instructions, context, query) triple and return the
    tokenized input along with the token indices for each section.

    Args:
        instructions (str): The instruction string.
        context (str): The context string.
        query (str): The query string.
        tokenizer (Callable): HuggingFace tokenizer.

    Returns:
        (tokenized_chat, instruction_tokens, context_tokens, query_tokens)
    """
    chat = [
        {"role": "user", "content": f"## Instructions:\n{instructions}\n\n## Context:\n{context}\n\n## Query:\n{query}"},
    ]
    formatted_chat = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )
    tokenized_chat = tokenizer(
        formatted_chat, return_offsets_mapping=True, add_special_tokens=False
    )

    instruction_start = formatted_chat.index("## Instructions:\n") + len("## Instructions:\n")
    instruction_end = formatted_chat.index("\n\n## Context:")
    context_start = formatted_chat.index("## Context:\n") + len("## Context:\n")
    context_end = formatted_chat.index("\n\n## Query:")
    query_start = formatted_chat.index("## Query:\n") + len("## Query:\n")
    query_end = query_start + len(query)

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


class JudgementLM:
    """
    A wrapper around NNsight language models that provides methods for generating text
    and caching output from attention activations.

    Args:
        model_name (str): The name of the model to load from NNsight or huggingface.
        sampling_params (dict): A dictionary of sampling parameters to pass to the
            NNsight LanguageModel generate method. Default is {}.
        nnsight_kwargs (dict): Additional keyword arguments to pass to the NNsight LanguageModel.
        verbose (bool): Whether to print verbose output during generation. Default is False.
    """
    def __init__(
        self,
        model_name : str,
        sampling_params : dict = {},
        nnsight_kwargs : dict = {},
        verbose : bool = False,
    ):
        self.model_name = model_name
        self.sampling_params = {'max_new_tokens': 50} | sampling_params
        self.max_new_tokens = self.sampling_params['max_new_tokens']
        self.verbose = verbose

        # Detect available GPUs and set up device allocation
        self._setup_devices()
        
        self.llm = StandardizedTransformer(model_name, enable_attention_probs=False, **nnsight_kwargs)
        print(self.llm)
        self.tokenizer = self.llm.tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.n_layers = len(self.llm.model.layers)
        self.n_heads = self.llm.config.num_attention_heads
        self.n_kv_heads = self.llm.config.num_key_value_heads
        self.head_dim = self.llm.config.hidden_size // self.n_heads

        self.responses = []
        self.parametric_score_arrays = []
        self.context_score_array = []
    

    def _setup_devices(self):
        """
        Set up device allocation for LLM and tensors.
        If multiple GPUs are available, use separate devices for LLM and tensors.
        Otherwise, use the same device for both.
        """
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            if (num_gpus >= 2):
                # Use different GPUs for LLM and tensors
                self.llm_device = torch.device("cuda:0")
                self.tensor_device = torch.device("cuda:1")
                if self.verbose:
                    print(f"Using {num_gpus} GPUs: LLM on cuda:0, tensors on cuda:1")
            else:
                # Single GPU: use same device for both
                self.llm_device = torch.device("cuda:0")
                self.tensor_device = torch.device("cuda:0")
                if self.verbose:
                    print(f"Using single GPU: cuda:0")
        else:
            # CPU fallback
            self.llm_device = torch.device("cpu")
            self.tensor_device = torch.device("cpu")
            if self.verbose:
                print("No GPU available, using CPU")

        #self.tensor_device = torch.device("cpu")
    

    def generate(
        self,
        instructions: str,
        context: str,
        query: str,
    ) -> dict[str, str | float]:
        """
        Generate a response for a single (instructions, context, query) triple.

        Runs the model under an NNsight trace to capture per-layer, per-head
        attention output projections at each generation step, then decodes the
        response.

        Args:
            instructions (str): The instructions string.
            context (str): The context string.
            query (str): The query string.

        Returns:
            dict: A dictionary containing:
                'response' (str): The decoded generated text.
                'logprob' (float): Summed log-probability of the generated tokens.
                'attn_output' (np.ndarray): Attention output for the last generated
                    token, shape (n_layers, n_heads, head_dim).
        """
        (tokenized_prompt,
         instruction_token_indices,
         context_token_indices,
         query_token_indices) = tokenize(
            instructions, context, query, self.tokenizer
        )
        prompt_len = len(tokenized_prompt)

        llm_device = self.llm.device
        tensor_device = self.tensor_device
        response_dict: dict[str, str | float] = {}

        with self.llm.generate(tokenized_prompt, **self.sampling_params) as tracer:
            attention_outputs = torch.zeros(
                size=(self.max_new_tokens, self.n_layers, self.n_heads, self.head_dim),
                device=tensor_device,
                dtype=torch.bfloat16,
            ).save()

            # Keep tokens/logprobs on CPU to avoid keeping large GPU graphs alive.
            response_tokens_cpu = torch.full(
                size=(self.max_new_tokens,),
                fill_value=self.tokenizer.pad_token_id,
                device=torch.device("cpu"),
                dtype=torch.long,
            ).save()

            response_log_prob = np.array([0.0], dtype=np.float64).save()

            with tracer.iter[:] as token_idx:
                for layer_idx, layer in enumerate(self.llm.model.layers):
                    # Output of attention is the input to the MLP
                    #print("Storing attention output for layer", layer_idx)
                    #print(layer.self_attn.o_proj.input[-1, -1, :].shape)
                    attention_outputs[token_idx, layer_idx, :, :] = (
                        layer.self_attn.o_proj.input[-1, -1, :]
                        .view(self.n_heads, self.head_dim)
                        .detach()
                        .to(tensor_device)
                    )
                
                logits_last = self.llm.logits[0, -1, :]
                tok_id = int(torch.argmax(logits_last).item())
                tok_logp = (logits_last[tok_id] - torch.logsumexp(logits_last, dim=-1)).item()

                response_tokens_cpu[token_idx] = tok_id
                response_log_prob[0] += float(tok_logp)

        response = self.llm.tokenizer.decode(response_tokens_cpu, skip_special_tokens=True)
        n_generated_tokens = (response_tokens_cpu != self.tokenizer.pad_token_id).sum().item()

        response_dict = {
            "response": response,
            "logprob": float(response_log_prob[0]),
        }
        response_dict["attn_output"] = attention_outputs[n_generated_tokens - 1].float().cpu().numpy()

        # Explicitly delete large tensors to free memory after .save() references
        del attention_outputs
        del response_tokens_cpu
        del tracer

        # Clear CUDA cache if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return response_dict

    
    def predict(
        self,
        prompts : list[tuple[str, str, str]],
    ) -> tuple[list[str], list[float]]:
        """
        Run generate() on a batch of (instructions, context, query) triples.

        Args:
            prompts (list[tuple[str, str, str]]): A list of (instructions, context, query) triples.

        Returns:
            list[dict]: One response dict per prompt. See generate() for the dict structure.
        """
        responses = []
        for i, (instructions, context, query) in enumerate(tqdm(prompts)):
            response_dict = self.generate(instructions, context, query)
            responses.append(response_dict)

        return responses



