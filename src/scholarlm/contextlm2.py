import os
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


class ContextLM2:
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
        verbose : bool = False,
        #cache_output_dir : str = None
    ):
        self.model_name = model_name
        self.top_k = top_k
        self.sampling_params = {'max_new_tokens': 50} | sampling_params
        self.max_new_tokens = self.sampling_params['max_new_tokens']
        self.return_full_output = return_full_output
        self.verbose = verbose
        #self.cache_output_dir = cache_output_dir
        #if self.cache_output_dir is not None:
        #    os.makedirs(self.cache_output_dir, exist_ok=True)

        self.llm = StandardizedTransformer(model_name, enable_attention_probs=True, **nnsight_kwargs)
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
    

    def compute_external_context_score(
        self,
        response_embeddings : torch.Tensor,
        context_embeddings : torch.Tensor,
        context_indices : torch.Tensor,
        attention_probabilities : list[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute the external context score as the cosine similarity between the last token embedding
        and the mean of the top-k context embeddings.

        Args:
            last_token_emb (torch.Tensor): The embeddings of the generated response.
            context_emb_cache (torch.Tensor): The cached embeddings of the context tokens.
            attention_probabilities (list[torch.Tensor]): A list of attention probability tensors
                from each layer of the model.

        Returns:
            torch.Tensor: A tensor of shape [num_layers, num_heads] containing the external context scores.
        """
        k = min(self.top_k, len(context_embeddings))
        external_context_scores = torch.zeros(
            (len(response_embeddings), self.n_layers, self.n_heads)
        )

        for token_idx, attn_probs in enumerate(attention_probabilities):
            for layer_idx in range(self.n_layers):
                A = attn_probs[layer_idx]
                for head_idx in range(self.n_heads):
                    attn_weights = A[head_idx, context_indices]  # Shape: [seq_len]
                    top_k_indices = torch.topk(attn_weights, k).indices.cpu()  # Indices of top-k context tokens
                    top_k_emb = context_embeddings[top_k_indices]  # Shape: [k, hidden_size]
                    mean_top_k_emb = torch.mean(top_k_emb, dim=0)  # Shape: [hidden_size]
                    cosine_similarity = F.cosine_similarity(
                        mean_top_k_emb,
                        response_embeddings[token_idx],
                        dim=-1
                    )
                    external_context_scores[token_idx, layer_idx, head_idx] = cosine_similarity

        return external_context_scores


    def compute_parametric_knowledge_score(
        self,
        mlp_inputs : torch.Tensor,
        mlp_outputs : torch.Tensor
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
        input_logits = self.llm.lm_head(self.llm.model.norm(mlp_inputs))
        output_logits = self.llm.lm_head(self.llm.model.norm(mlp_outputs))

        # Convert logits to probabilities
        input_probs = torch.nn.functional.softmax(input_logits, dim=-1).detach().cpu().numpy()
        output_probs = torch.nn.functional.softmax(output_logits, dim=-1).detach().cpu().numpy()
        jsd = jensenshannon(input_probs, output_probs, axis = -1)

        return jsd
    

    def compute_copying_score(
        self,
        layer_inputs : torch.Tensor,
        context_token_indices : torch.Tensor,
        attention_probabilities : list[list[torch.Tensor]]
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
        copying_scores = torch.zeros((self.n_layers, self.n_heads))
        for layer_idx, layer in enumerate(self.llm.model.layers):
            out_weights = layer.self_attn.o_proj.weight # Shape: [D, H * D_head]
            value_weights = layer.self_attn.v_proj.weight # Shape: [H_kv * D_head, D]

            A = attention_probabilities[0][layer_idx].cuda() # Shape: [H, N] (last token attentions at each head)
            X = layer_inputs[layer_idx, :, :] # Shape: [N, D]
            XC = layer_inputs[layer_idx, context_token_indices, :] # Shape: [N_ctx, D]

            for head_idx in range(self.n_heads):
                # Self attention
                a = A[head_idx, :]
                Xa = torch.matmul(X.transpose(0,1), a) # Shape: [D]
                XCa = torch.matmul(XC.transpose(0,1), a[context_token_indices]) # Shape: [D]

                # OV matrix
                kv_head_idx = head_idx % self.n_kv_heads
                O = out_weights[:, head_idx * self.head_dim : (head_idx + 1) * self.head_dim] # Shape: [D, D_head]
                V = value_weights[kv_head_idx * self.head_dim : (kv_head_idx + 1) * self.head_dim, :] # Shape: [D_head, D]
                W_OV = torch.matmul(O, V) # Shape: [D, D]

                # Compare full head output to self-attended context
                W_OV_Xa = torch.matmul(W_OV, Xa) # Shape: [D]
                copying_score = F.cosine_similarity(W_OV_Xa, XCa, dim=0)
                copying_scores[layer_idx, head_idx] = copying_score
                
        return copying_scores
    

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
        prompt_len = len(tokenized_prompt)
        k = min(self.top_k, len(context_token_indices))

        with self.llm.generate(tokenized_prompt, **self.sampling_params) as tracer:
            layer_inputs = torch.zeros(
                self.n_layers, prompt_len, self.llm.config.hidden_size, device=self.llm.device
            ).save()

            attention_probabilities = (
                [[None for _ in range(len(self.llm.model.layers))] for _ in range(self.max_new_tokens)]
            ).save()

            mlp_inputs = torch.zeros(
                size = (self.max_new_tokens, self.n_layers, self.llm.config.hidden_size), device=self.llm.device
            ).save()

            mlp_outputs = torch.zeros(
                size = (self.max_new_tokens, self.n_layers, self.llm.config.hidden_size), device=self.llm.device
            ).save()

            context_embeddings = torch.zeros(
                (len(context_token_indices), self.llm.config.hidden_size)
            ).save()

            response_embeddings = torch.zeros(
                (self.max_new_tokens, self.llm.config.hidden_size)
            ).save()

            response_tokens = torch.full(
                size = (self.max_new_tokens,),
                fill_value = self.tokenizer.pad_token_id # Fill with pad token initially
            ).save()

            with tracer.iter[:] as token_idx:
                for layer_idx, layer in enumerate(self.llm.model.layers):
                    # Cache layer inputs
                    layer_input = self.llm.layers_input[layer_idx]
                    if token_idx == 0:
                        layer_inputs[layer_idx, :, :] = layer_input
                    
                    # Attention shape: [batch_size, num_heads, seq_len, seq_len]
                    attention_probabilities[token_idx][layer_idx] = self.llm.attention_probabilities[layer_idx][-1,:,-1,:].cpu()
                    
                    # Cache MLP inputs and outputs for this layer
                    mlp_inputs[token_idx, layer_idx, :] = self.llm.mlps_input[layer_idx][-1, -1, :]
                    mlp_outputs[token_idx, layer_idx, :] = self.llm.mlps_output[layer_idx][-1, -1, :]

                # Last layer embeddings for context and current token:
                if token_idx == 0:
                    context_embeddings[:,:] = self.llm.model.output.last_hidden_state[-1, context_token_indices, :]

                response_embeddings[token_idx, :] = self.llm.model.output.last_hidden_state[-1, -1, :]
                
                # Compute response tokens:
                response_tokens[token_idx] = self.llm.logits[0, -1, :].argmax().save()


        response = self.llm.tokenizer.decode(response_tokens.cpu(), skip_special_tokens=True)
        response_dict = {
            "response": response,
        }

        n_generated = np.sum(response_tokens.cpu().numpy() != self.tokenizer.pad_token_id)

        if self.return_full_output:
            # Linear probe
            response_dict['linear_probes'] = response_embeddings[:n_generated, :].cpu().numpy()

            # External context scores
            response_dict['context_scores'] = self.compute_external_context_score(
                response_embeddings[:n_generated, :],
                context_embeddings,
                context_token_indices,
                attention_probabilities[:n_generated]
            ).cpu().numpy()

            # Parametric knowledge scores
            response_dict['parametric_scores'] = self.compute_parametric_knowledge_score(
                mlp_inputs[:n_generated, :, :],
                mlp_outputs[:n_generated, :, :]
            )

            # Copying scores
            response_dict['copying_scores'] = self.compute_copying_score(
                layer_inputs,
                context_token_indices,
                attention_probabilities,
            ).detach().cpu().numpy()

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
        

        
    