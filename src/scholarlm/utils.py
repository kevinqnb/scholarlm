import os
import base64
from io import BytesIO
import pandas as pd
import torch
from typing import Callable
from PIL import Image
import pytesseract


####################################################################################################


def get_filenames_in_directory(
    directory_path : str, ignore : list[str] = None
) -> list[str]:
    """
    Returns a list of all filenames in the specified directory.
    
    Args:
        directory_path (str): The path to the directory.

        ignore (List[str]): A list of filenames to ignore. Defaults to None.
    
    Returns:
        list: A list of filenames in the directory.
    """
    try:
        filenames = [
            f for f in os.listdir(directory_path) 
            if os.path.isfile(os.path.join(directory_path, f)) and (ignore is None or f not in ignore)
        ]
        return filenames
    except FileNotFoundError:
        return f"Error: Directory not found: {directory_path}"
    except NotADirectoryError:
         return f"Error: Not a directory: {directory_path}"
    

####################################################################################################


def get_foldernames_in_directory(
    directory_path : str, ignore : list[str] = None
) -> list[str]:
    """
    Returns a list of all folder names in the specified directory.
    
    Args:
        directory_path (str): The path to the directory.

        ignore (List[str]): A list of folder names to ignore. Defaults to None.
    
    Returns:
        list: A list of folder names in the directory.
    """
    try:
        foldernames = [
            f for f in os.listdir(directory_path) 
            if os.path.isdir(os.path.join(directory_path, f)) and (ignore is None or f not in ignore)
        ]
        return foldernames
    except FileNotFoundError:
        return f"Error: Directory not found: {directory_path}"
    except NotADirectoryError:
         return f"Error: Not a directory: {directory_path}"


####################################################################################################


def encode_pil_image(pil_image):
    """
    Encode a PIL image to a base64 string.

    Args:
        pil_image (PIL.Image): The PIL image to encode. 

    Returns:
        str: The base64 encoded string of the image.
    """
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


####################################################################################################

def tokenize(
    instructions : str,
    context : str,
    query : str,
    tokenizer: callable,
) -> tuple[list[int], list[int], list[int]]:
    """
    Apply a chat template to a (context, instructions) pair, and return the tokenized input
    along with the indices of the tokens corresponding to context and instruction text.

    Args:
        instructions (str): The instruction string.
        context (str): The context string.
        query (str): The query string.
        tokenizer (Callable): Huggingface tokenizer.

    Returns:
        tokenized_chat, context_tokens, prompt_tokens (tuple[list[int] * 3]): A tuple containing:
            1. The tokenized input represented as a list of integer token ids.
            2. A list of indices from tokenized_chat corresponding to the instructions.
            3. A list of indices from tokenized_chat corresponding to the context.
            4. A list of indices from tokenized_chat corresponding to the query.
    """
    chat = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": f"## Context:\n{context}\n\n## Query:\n{query}"},
    ]
    formatted_chat = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )
    tokenized_chat = tokenizer(
        formatted_chat, return_offsets_mapping=True, add_special_tokens=False
    )

    instruction_start, instruction_end = (
        formatted_chat.index(instructions), formatted_chat.index(instructions) + len(instructions)
    )
    context_start, context_end = (
        formatted_chat.index(context), formatted_chat.index(context) + len(context)
    )
    query_start, query_end = (
        formatted_chat.index(query), formatted_chat.index(query) + len(query)
    )

    instruction_tokens = [
        i for i, (s, e) in enumerate(tokenized_chat["offset_mapping"])
        if s >= instruction_start and e <= instruction_end
    ]
    context_tokens = [
        i for i, (s, e) in enumerate(tokenized_chat["offset_mapping"])
        if s >= context_start and e <= context_end
    ]
    query_tokens  = [
        i for i, (s, e) in enumerate(tokenized_chat["offset_mapping"])
        if s >= query_start and e <= query_end
    ]

    return tokenized_chat["input_ids"], instruction_tokens, context_tokens, query_tokens


####################################################################################################


def jensen_shannon_divergence(
    p: torch.Tensor, q: torch.Tensor
) -> torch.Tensor:
    """
    Compute JSD(P||Q) for batches of distribution sequences. Specifically, this assumes 
    p and q are of shape [m, n, d], where m is the batch size (number of trials), 
    n is the sequence length (number of pairs within a batch to compute JSD for),
    and d is the distribution dimension (number of classes).

    The output is a tensor of shape [m,n], where each entry (i,j) represents the JSD 
    the j-th pair of distributions from the i-th batch / trial.

    Args:
        p (torch.Tensor): Tensor of shape [m, n, d]
        q (torch.Tensor): Tensor of shape [m, n, d]
    Returns:
        jsd (torch.Tensor): JSD tensor of shape [m,n].
    """
    # Replace zeros with small value to avoid NaNs
    p = p.clamp(min=1e-10)
    q = q.clamp(min=1e-10)
    
    # Normalize (if not already normalized)
    p = p / p.sum(dim=-1, keepdim=True)
    q = q / q.sum(dim=-1, keepdim=True)
    
    m = 0.5 * (p + q)

    # Compute KL divergences
    kl_pm = (p * (p / m).log()).sum(dim=-1)
    kl_qm = (q * (q / m).log()).sum(dim=-1)
    
    jsd = 0.5 * (kl_pm + kl_qm)
    return jsd


####################################################################################################


def correct_image_orientation(pil_image):
    """
    Detects orientation of a PIL image using Tesseract OSD and returns a rotated image corrected to upright.

    Args:
        pil_image (PIL.Image.Image): Input image

    Returns:
        PIL.Image.Image: Upright-corrected image
    """
    osd_output = pytesseract.image_to_osd(pil_image)
    
    # Parse the rotation angle from OSD output
    rotate_angle = 0
    for line in osd_output.splitlines():
        if "Rotate" in line:
            rotate_angle = int(line.split(":")[-1].strip())
            break

    if rotate_angle == 0:
        return pil_image
    else:
        corrected_image = pil_image.rotate(-rotate_angle, expand=True)
        return corrected_image
    

####################################################################################################