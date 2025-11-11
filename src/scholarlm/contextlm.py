from tqdm import tqdm
import math
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from nnsight import LanguageModel
from .utils import tokenize, jensen_shannon_divergence


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Helper function to rotate the last dimension of a tensor by half.

    Note: This function is sourced from the following open source repository:
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L178


    Args:
        x (torch.Tensor): The input tensor to rotate.
    Returns:
        (torch.Tensor): The rotated tensor.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1
):
    """
    Applies Rotary Position Embedding to the query and key tensors.

    Note: This function is sourced from the following open source repository:
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L178


    Args:
        q (torch.Tensor): The query tensor.
        k (torch.Tensor): The key tensor.
        cos (torch.Tensor): The cosine part of the rotary embedding.
        sin (torch.Tensor): The sine part of the rotary embedding.
        unsqueeze_dim (int): Specifies the dimension along which to unsqueeze cos and
            sin so that they can be properly broadcasted to the dimensions of q and k.
            For example, suppose that cos and sin have the shape 
            [batch_size, seq_len, head_dim]. Then, if q and k have the shape [batch_size, heads,
            seq_len, head_dim], setting unsqueeze_dim=1 makes cos and
            sin broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        tuple(torch.Tensor): comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class ContextLM:
    """
    A wrapper around NNsight language models that provides methods for generating text
    and computing hallucination scores based upon input context and instructions.

    This is intended to be an application of methods described in the following paper:
    Sun, Zhongxiang, et al. "ReDeEP: Detecting Hallucination in Retrieval-Augmented Generation
    via Mechanistic Interpretability." ICLR. 2025.

    Args:
        model_name (str): The name of the model to load from NNsight or huggingface.
        top_k (float): The fraction of context tokens with largest attention weight to 
            compare generated tokens with (for external context score). Default is 0.1 (10%).
        sampling_params (dict): A dictionary of sampling parameters to pass to the
            NNsight LanguageModel generate method. Default is {}.
        nnsight_kwargs (dict): Additional keyword arguments to pass to the NNsight LanguageModel.
        return_full_output (bool): Whether to return full per-layer and per-head scores
            in the output dictionary. Default is False.
        verbose (bool): Whether to print verbose output during generation. Default is False.
    """
    def __init__(
        self,
        model_name : str,
        top_k : int = 10,
        sampling_params : dict = {},
        nnsight_kwargs : dict = {},
        return_full_output : bool = False,
        verbose : bool = False
    ):
        self.model_name = model_name
        self.top_k = top_k
        self.sampling_params = {'max_new_tokens': 50} | sampling_params
        self.max_new_tokens = self.sampling_params['max_new_tokens']
        self.return_full_output = return_full_output
        self.verbose = verbose

        self.llm = LanguageModel(model_name, **nnsight_kwargs)
        print(self.llm)
        self.tokenizer = self.llm.tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.n_layers = len(self.llm.model.layers)
        self.n_heads = self.llm.config.num_attention_heads
        self.head_dim = self.llm.config.hidden_size // self.n_heads

        self.responses = []
        self.parametric_score_arrays = []
        self.context_score_array = []
    

    def compute_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        key_cache: torch.Tensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention weights for the last query token against all key tokens,
        using cached key states for previous tokens.

        Note: This code is adapted from the following open source repository:
        https://github.com/sfeucht/dual-route-induction/blob/2743c6117b973aba5661e1d286c90342c4ff7c4a/scripts/utils.py

        Args:
            query_states (torch.Tensor): The query states of shape (batch_size, seq_len, query_hidden_dim).
            key_states (torch.Tensor): The key states of shape (batch_size, seq_len, key_hidden_dim).
            key_cache (torch.Tensor | None): The cached key states of shape 
                (batch_size, num_heads, cached_seq_len, head_dim), or None if no cache.
            position_embeddings (tuple[torch.Tensor, torch.Tensor]): Tuple of (cosine, sine)
                positional embeddings.
            attention_mask (torch.Tensor | None): The attention mask of shape 
                (batch_size, 1, seq_len, total_seq_len), or None if no mask.
        Returns:
            attn_weights (torch.Tensor): The attention weights of shape 
                (batch_size, num_heads, 1, total_seq_len).
            updated_key_cache (torch.Tensor): The updated key cache of shape 
                (batch_size, num_heads, total_seq_len, head_dim).
        """
        bsz = query_states.shape[0]; seq_len = query_states.shape[1]
        query_states = query_states.view(bsz, seq_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, seq_len, -1, self.head_dim).transpose(1, 2)

        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, position_embeddings[0], position_embeddings[1]
        )

        # If key,value matrices are grouped, repeat to match number of heads
        key_states = key_states.repeat_interleave(self.n_heads // key_states.shape[1], dim=1)

        if key_cache is None:
            updated_key_cache = key_states
        else:
            updated_key_cache = torch.cat((key_cache, key_states), dim=2)

        # Always use the last query (the new token)
        last_query = query_states[:, :, -1:, :]

        # Compute attention against cached keys
        attn_weights = torch.matmul(last_query, updated_key_cache.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None: # NOTE: I think this is always None for the situations I'm using.... but good to double check.
            causal_mask = attention_mask[:, :, -1, : updated_key_cache.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = F.softmax(attn_weights, dim=-1)

        return attn_weights, updated_key_cache
    

    def compute_external_context_score(
        self,
        last_token_emb : torch.Tensor,
        context_emb_cache : torch.Tensor,
        context_top_indices : torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the external context score as the cosine similarity between the last token embedding
        and the mean of the top-k context embeddings.

        Args:
            last_token_emb (torch.Tensor): The embedding of the last generated token.
            context_emb_cache (torch.Tensor): The cached embeddings of the context tokens.
            context_top_indices (torch.Tensor): The indices of the top-k context tokens for each head.

        Returns:
            torch.Tensor: A tensor of shape [num_layers, num_heads] containing the external context scores.
        """        
        n_layers = len(self.llm.model.layers)
        top_k_indices = context_top_indices[:, 0, :, 0, :].flatten(start_dim = 0, end_dim = 1)
        flattened_indices = top_k_indices.flatten() 
        top_k_emb = context_emb_cache[flattened_indices]
        top_k_emb = top_k_emb.view(top_k_indices.shape[0], top_k_indices.shape[1], -1)
        mean_top_k_emb = torch.mean(top_k_emb, dim=1)
        cosine_similarity = F.cosine_similarity(mean_top_k_emb, last_token_emb, dim=1)
        external_context_scores = cosine_similarity.view(n_layers, self.n_heads)

        return external_context_scores


    def compute_parametric_knowledge_score(
        self,
        mlp_input : torch.Tensor,
        mlp_output : torch.Tensor
    ) -> float:
        """
        Compute the parametric knowledge score as the Jensen-Shannon Divergence between
        the MLP input and output distributions.

        Args:
            mlp_input (torch.Tensor): The input to the MLP layer (before transformation).
            mlp_output (torch.Tensor): The output from the MLP layer (after transformation).

        Returns:
            float: The computed parametric knowledge score.
        """
        # Calculate logits for the last token before and after MLP
        input_logits = self.llm.lm_head(self.llm.model.norm(mlp_input[:, -1, :]))
        output_logits = self.llm.lm_head(self.llm.model.norm(mlp_output[:, -1, :]))

        # Convert logits to probabilities
        input_probs = torch.nn.functional.softmax(input_logits, dim=-1)
        output_probs = torch.nn.functional.softmax(output_logits, dim=-1)

        return jensen_shannon_divergence(input_probs, output_probs)
    

    def compute_copying_score(
        self,
        attn_weights: torch.Tensor,
        out_weights: torch.Tensor,
        value_weights: torch.Tensor,
        token_hidden_states: torch.Tensor, 
        context_token_indices: torch.Tensor
    ):
        """
        Compute a copying score based on attention weights, overlap matrix, and token hidden states.

        Args:
            attn_weights (torch.Tensor): The attention weights tensor.
            ov_matrix (torch.Tensor): The overlap matrix tensor.
            token_hidden_states (torch.Tensor): The token hidden states tensor.
        Returns:
            float: The computed copying score.
        """
        out_weights = out_weights.view(-1, self.head_dim, self.llm.config.hidden_size).transpose(1, 2)
        value_weights = value_weights.view(-1, self.llm.config.hidden_size, self.head_dim).transpose(1, 2)
        value_weights = value_weights.repeat_interleave(self.n_heads // value_weights.shape[0], dim=0)
        W_OV = torch.matmul(out_weights, value_weights)

        token_hidden_states = token_hidden_states[:, :attn_weights.shape[-1], :].transpose(1,2)
        attn_weights = attn_weights[-1,:,:,:].transpose(1,2)
        v = torch.matmul(token_hidden_states, attn_weights)
        vc = torch.matmul(token_hidden_states[:, :, context_token_indices], attn_weights[:, context_token_indices, :])
        Wv = torch.matmul(W_OV, v)
        copying_score = F.cosine_similarity(Wv, vc, dim=1)
        copying_score = copying_score[:, -1]
        return copying_score
    

    def generate(
        self,
        instructions: str,
        context: str,
        query: str
    ) -> dict[str, str | float]:
        """
        Generate text for a (context, instructions) pair, and compute
        external context scores and parametric knowledge scores for each generated token.

        Args:
            instructions (str): The instructions string.
            context (str): The context string.
            query (str): The query string.

        Returns:
            response_dict (dict): A dictionary containing:
                'response' (str): The generated text.
                'parametric_score' (float): The summed parametric knowledge score.
                'context_score' (float): The summed external context score.
        """
        (tokenized_prompt,
         instruction_token_indices,
         context_token_indices,
         query_token_indices) = tokenize(
            instructions, context, query, self.tokenizer
        )
        #k = math.ceil(self.top_k * len(context_token_indices))
        prompt_len = len(tokenized_prompt)
        k = min(self.top_k, len(context_token_indices))
        with self.llm.generate(tokenized_prompt, **self.sampling_params) as tracer:
            # Cache key matrices and context embeddings to use for external context score computation
            key_cache = [None] * len(self.llm.model.layers)

            context_top_indices = torch.zeros(
                (self.max_new_tokens, self.n_layers, 1, self.n_heads, 1, k),
                dtype = torch.long
            ).save()

            context_emb = torch.zeros(
                (len(context_token_indices), self.llm.config.hidden_size)
            ).save()

            layer_inputs = torch.zeros(
                self.n_layers, prompt_len + self.max_new_tokens - 1, self.llm.config.hidden_size
            ).save()

            # Record external context scores
            external_context_scores = torch.zeros(
                (self.max_new_tokens, self.n_layers, self.n_heads)
            ).save()

            # Record parametric knowledge scores
            parametric_knowledge_scores = torch.zeros(
                (self.max_new_tokens, self.n_layers)
            ).save()

            # Record copying scores
            copying_scores = torch.zeros(
                (self.max_new_tokens, self.n_layers, self.n_heads)
            ).save()

            # Record layer hidden states
            layer_hidden_states = torch.zeros(
                (self.max_new_tokens, self.n_layers, self.llm.config.hidden_size)
            ).save()

            # Record the response tokens
            response_tokens = torch.full(
                size = (self.max_new_tokens,),
                fill_value = self.tokenizer.pad_token_id # Fill with pad token initially
            ).save()

            with tracer.iter[:] as token_idx:
                if self.return_full_output:
                    for layer_idx, layer in enumerate(self.llm.model.layers):
                        layer_input = layer.input[-1, :, :]
                        
                        if token_idx == 0:
                            layer_inputs[layer_idx, :prompt_len, :] = layer_input
                        else:
                            layer_inputs[layer_idx, prompt_len + token_idx - 1: prompt_len + token_idx, :] = layer_input

                        # Compute attention weights:
                        position = layer.self_attn.inputs[1]['position_embeddings']
                        attention_mask = layer.self_attn.inputs[1]['attention_mask']

                        query_states = layer.self_attn.q_proj.output
                        key_states = layer.self_attn.k_proj.output
                        out_weights = layer.self_attn.o_proj.weight
                        value_weights = layer.self_attn.v_proj.weight

                        attn_weights, updated_key_cache = self.compute_attention(
                            query_states, key_states, key_cache[layer_idx], position, attention_mask
                        )
                        key_cache[layer_idx] = updated_key_cache

                        # Find top-k context indices for each head
                        context_attn_weights = attn_weights[:, :, :, context_token_indices]

                        layer_context_top_values, layer_context_top_indices = torch.topk(
                            context_attn_weights, k=k, dim=-1
                        )
                        context_top_indices[token_idx, layer_idx, :, :, :, :] = layer_context_top_indices

                        # Compute parametric knowledge score
                        parametric_knowledge_scores[token_idx, layer_idx] = self.compute_parametric_knowledge_score(
                            layer.mlp.input, layer.mlp.output
                        )

                        out_weights = layer.self_attn.o_proj.weight
                        value_weights = layer.self_attn.v_proj.weight

                        # Compute copying score
                        copying_scores[token_idx, layer_idx, :] = self.compute_copying_score(
                            attn_weights = attn_weights,
                            out_weights = out_weights,
                            value_weights = value_weights,
                            token_hidden_states = layer_inputs,
                            context_token_indices = context_token_indices
                        )
                        

                        # Record layer response hidden states
                        layer_hidden_states[token_idx, layer_idx, :] = layer.output[:, -1, :]

                    
                    # Last layer embeddings for context and current token:
                    if token_idx == 0:
                        context_emb[:,:] = self.llm.model.output.last_hidden_state[:, context_token_indices, :]
                    last_token_emb = self.llm.model.output.last_hidden_state[:, -1, :].cpu()

                    # Compute external context score:
                    external_context_scores[token_idx, :, :] = self.compute_external_context_score(
                        last_token_emb, context_emb, context_top_indices[token_idx, :, :, :, :, :]
                    )

                # Compute response tokens:
                response_tokens[token_idx] = self.llm.output["logits"][0, -1, :].argmax(dim=-1)


        response = self.llm.tokenizer.decode(response_tokens.cpu(), skip_special_tokens=True)

        if self.return_full_output:
            # Average scores over response tokens:
            avg_parametric_knowledge_scores = parametric_knowledge_scores.mean(dim=0)
            avg_parametric_knowledge_scores = avg_parametric_knowledge_scores.cpu().detach().numpy()
            self.parametric_score_arrays.append(avg_parametric_knowledge_scores)
            avg_external_context_scores = external_context_scores.mean(dim=0)
            avg_external_context_scores = avg_external_context_scores.cpu().detach().numpy()
            self.context_score_array.append(avg_external_context_scores)

            # Get last token hidden states
            layer_hidden_states = layer_hidden_states.cpu().detach().numpy()
            last_token_hidden_states = layer_hidden_states[-1, :, :]

            parametric_score_dict = {
                f"parametric_l{layer_idx}": avg_parametric_knowledge_scores[layer_idx]
                for layer_idx in range(self.n_layers)
            }
            external_context_score_dict = {
                f"context_l{layer_idx}h{head_idx}": avg_external_context_scores[layer_idx, head_idx]
                for layer_idx, head_idx in 
                [(l, h) for l in range(self.n_layers) for h in range(self.n_heads)]
            }
            linear_probe_dict = {
                f"probe_l{layer_idx}": last_token_hidden_states[layer_idx]
                for layer_idx in range(self.n_layers)
            }
            copying_score_dict = {
                f"copying_l{layer_idx}h{head_idx}": copying_scores[-1, layer_idx, head_idx].cpu().detach().numpy()
                for layer_idx, head_idx in 
                [(l, h) for l in range(self.n_layers) for h in range(self.n_heads)]
            }
            response_dict = {
                "response": response,
                "parametric_score": parametric_score_dict,
                "context_score": external_context_score_dict,
                "linear_probe": linear_probe_dict,
                "copying_score": copying_score_dict
            }
        else:
            response_dict = {
                "response": response,
            }

        return response_dict

    
    def predict(
        self,
        prompts : list[tuple[str, str, str]]
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
        for instructions, context, query in tqdm(prompts):
            response_dict = self.generate(instructions, context, query)
            responses.append(response_dict)
            self.responses.append(response_dict)

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
        

        
    