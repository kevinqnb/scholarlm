import os
from dotenv import load_dotenv
load_dotenv()

import torch

# import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookPoint,
)  # Hooking utilities
from transformer_lens import HookedTransformer, FactoredMatrix

#from scholarlm import ContextLM, jensen_shannon_divergence


####################################################################################################
# Load model:
# "meta-llama/Llama-3.1-8B-Instruct"

device = utils.get_device()
print(f"Using device: {device}")

model = HookedTransformer.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct", device=device
)

prompts = [
    ("Use the context to directly answer the given query. Do not include any other text or punctuation.", "The latitude of Paris, France recently changed and is now 52.1375.", "What is the latitude of Paris, France?"),
    #("Use the context to directly answer the given query. Do not include any other text or punctuation.", "The color of the sky is purple today.", "What is the color of the sky?"),
]

model_tokens = model.to_tokens(prompts[0][0] + " " + prompts[0][1] + " " + prompts[0][2])

mlp_in = torch.zeros(
    (model.cfg.n_layers, model.cfg.d_model),
    device=device,
)

def mlp_in_hook(
    hidden_states: torch.Tensor, # [batch, seq_len, d_model]
    hook: HookPoint
):
    print(f"Hooking layer {hook.layer()}")
    print(f"Hidden states shape: {hidden_states.shape}")
    mlp_in[hook.layer(), :] = hidden_states[-1,-1,:]

mlp_in_hook_names_filter = lambda name: name.endswith("ln2.hook_normalized")

#model_logits, model_cache = model.run_with_cache(model_tokens)
model.run_with_hooks(
    model_tokens,
    fwd_hooks=[(mlp_in_hook_names_filter, mlp_in_hook) for layer in range(model.cfg.n_layers)],
)



####################################################################################################
