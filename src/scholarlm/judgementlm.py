import os
import gc
from tqdm import tqdm
import math
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from nnsight import LanguageModel
from nnterp import StandardizedTransformer
from .utils import tokenize, jensen_shannon_divergence
from scipy.spatial.distance import jensenshannon


class JudgementLM:
    """
    A wrapper around NNsight language models that provides methods for generating text
    and computing hallucination scores based upon input context and instructions.

    This is intended to be an application of methods described in the following paper:
    Sun, Zhongxiang, et al. "ReDeEP: Detecting Hallucination in Retrieval-Augmented Generation
    via Mechanistic Interpretability." ICLR. 2025.

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
        
        #self.llm = LanguageModel(model_name, **nnsight_kwargs)
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
        Generate text for a (context, instructions) pair, and compute
        external context scores and parametric knowledge scores for each generated token.

        Args:
            instructions (str): The instructions string.
            context (str): The context string.
            query (str): The query string.
            return_attn_output (bool): If True, also return per-layer attention output (can be large).

        Returns:
            response_dict (dict): A dictionary containing:
                'response' (str): The generated text.
                'logprob' (float): The summed log-probability of the selected tokens.
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
        Generate text for a batch of (context, instructions) pairs, and compute
        a hallucination score for each generation.

        Args:
            prompts (list[tuple[str, str, str]]): A list of (instructions, context, query) pairs.
        
        Returns:
            responses (list[dict]): A list of dictionaries containing:
                'response' (str): The generated text.
                'parametric_score' (float): The summed parametric knowledge score.
                'context_score' (float): The summed external context score.
        """
        responses = []
        for i, (instructions, context, query) in enumerate(tqdm(prompts)):
            response_dict = self.generate(instructions, context, query)
            responses.append(response_dict)

        return responses
    

    def save(
        self,
        path : str
    ):
        """
        Save the recorded responses, parametric scores, and context scores to a .npz file.

        Args:
            path (str): The file path to save the data to.
        """
        np.savez(
            path,
            responses = self.responses,
            parametric_scores = np.array(self.parametric_score_arrays),
            context_scores = np.array(self.context_score_array)
        )



