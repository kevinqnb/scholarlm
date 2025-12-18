import os
import base64
from io import BytesIO
import pandas as pd
import torch
from typing import Callable
from PIL import Image
import pytesseract
from bs4 import BeautifulSoup
import subprocess
from pypdf import PdfReader


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


def get_pdf_page_dimensions(pdf_path: str, page_num: int) -> tuple[float, float]:
    """
    Get PDF page dimensions in points using pdfinfo.
    
    Args:
        pdf_path: Path to PDF file
        page_num: Page number (1-indexed)
    
    Returns:
        Tuple of (width, height) in points
    """
    result = subprocess.run(
        ["pdfinfo", "-f", str(page_num), "-l", str(page_num), "-box", pdf_path],
        capture_output=True,
        text=True,
        timeout=30
    )
    
    if result.returncode != 0:
        raise ValueError(f"pdfinfo failed:  {result.stderr}")
    
    # Parse MediaBox from output
    for line in result.stdout.splitlines():
        if "MediaBox" in line:
            parts = line.split(":", 1)[1].strip().split()
            if len(parts) >= 4:
                x0, y0, x1, y1 = map(float, parts[:4])
                width = x1 - x0
                height = y1 - y0
                return width, height
    
    raise ValueError("MediaBox not found in PDF info")


####################################################################################################


def load_pdf_page(
    pdf_path: str,
    page_num: int,
    target_longest_dim: int = 2048
) -> Image.Image:
    """
    Render a PDF page to a high-quality PIL Image.
    
    Args:
        pdf_path: Path to PDF file
        page_num:  Page number (1-indexed)
        target_longest_dim: Target size for longest dimension in pixels
    
    Returns:
        PIL Image object
    """
    # Get page dimensions
    width, height = get_pdf_page_dimensions(pdf_path, page_num)
    longest_dim = max(width, height)
    
    # Calculate DPI needed to achieve target pixel dimension
    dpi = int(target_longest_dim * 72 / longest_dim)
    
    # Render PDF page to PNG using pdftoppm
    result = subprocess.run(
        [
            "pdftoppm",
            "-png",
            "-f", str(page_num),
            "-l", str(page_num),
            "-r", str(dpi),
            pdf_path
        ],
        capture_output=True,
        timeout=120
    )
    
    if result.returncode != 0:
        raise RuntimeError(f"pdftoppm failed: {result.stderr.decode()}")
    
    # Convert PNG bytes to PIL Image
    image = Image.open(BytesIO(result.stdout))
    
    # Ensure image is in RGB mode for VLM processing
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    return image


####################################################################################################


def correct_image_orientation(pil_image):
    """
    Detects orientation of a PIL image using Tesseract OSD and returns a rotated image corrected to upright.

    Args:
        pil_image (PIL.Image.Image): Input image

    Returns:
        PIL.Image.Image: Upright-corrected image
    """
    try:
        osd_output = pytesseract.image_to_osd(pil_image)
    
        # Parse the rotation angle from OSD output
        rotate_angle = 0
        for line in osd_output.splitlines():
            if "Rotate" in line:
                rotate_angle = int(line.split(":")[-1].strip())
                break

    except pytesseract.TesseractError as e:
        print(f"Tesseract OSD failed, proceeding without orientation correction.")
        print(e)
        print()
        rotate_angle = 0
    
    if rotate_angle == 0:
        return pil_image
    else:
        corrected_image = pil_image.rotate(-rotate_angle, expand=True)
        return corrected_image


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


def process_pdf(
    pdf_path: str,
    target_longest_dim: int = 2048
) -> list[str]:
    """
    Process all pages of a PDF with orientation correction. 
    Returns as a list of base64-encoded images for each page.
    
    Args:
        pdf_path: Path to PDF file
        target_longest_dim: Target size for longest dimension
    
    Returns:
        List of base64-encoded strings for each page
    """
    # Count pages
    reader = PdfReader(pdf_path)
    num_pages = len(reader.pages)

    results = []    
    for page_num in range(1, num_pages + 1):
        pil_image = load_pdf_page(pdf_path, page_num, target_longest_dim)
        pil_image = correct_image_orientation(pil_image)
        b64_image = encode_pil_image(pil_image)
        results.append(b64_image)
    
    return results


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
        #{"role": "system", "content": instructions},
        {"role": "user", "content": f"## Instructions:\n{instructions}\n\n## Context:\n{context}\n\n## Query:\n{query}"},
    ]
    formatted_chat = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )
    tokenized_chat = tokenizer(
        formatted_chat, return_offsets_mapping=True, add_special_tokens=False
    )

    '''
    instruction_start, instruction_end = (
        formatted_chat.index(instructions), formatted_chat.index(instructions) + len(instructions)
    )
    context_start, context_end = (
        formatted_chat.index(context), formatted_chat.index(context) + len(context)
    )
    query_start, query_end = (
        formatted_chat.index(query), formatted_chat.index(query) + len(query)
    )
    '''
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


def table_extract(
        table_df: pd.DataFrame,
        row_indices: list[str],
        column_indices: list[str]
    ) -> any:
    """
    Extract a value from a pandas DataFrame based on specified row and column indices.

    Args:
        table_df (pd.DataFrame): The DataFrame to extract the value from.
        row_indices (list[str]): A list of row index names to identify the target row.
        column_indices (list[str]): A list of column index names to identify the target column.
    Returns:
        any: The extracted value from the DataFrame.
    """
    flattened_table = table_df.reset_index(drop = True).T.reset_index().T.reset_index(drop=True)
    row_mask = ((flattened_table == val).sum(axis = 1) for i, val in enumerate(row_indices))
    row_mask = pd.concat(row_mask, axis=1).all(axis=1)
    row_idx = row_mask[row_mask].index[0]
    column_mask = ((flattened_table == val).sum(axis = 0) for i, val in enumerate(column_indices))
    column_mask = pd.concat(column_mask, axis=1).all(axis=1)
    column_idx = column_mask[column_mask].index[0]

    return flattened_table.iat[row_idx, column_idx]


####################################################################################################


def add_row_names(html_string):
    """
    Add row names (indices) to all non-header rows in an HTML table.
    
    Adds a <th> tag at the beginning of each row that doesn't already 
    contain <th> tags, numbering them sequentially starting from 1.
    
    Args:
        html_string: String containing HTML table
        
    Returns:
        String containing table with row names added
    """
    soup = BeautifulSoup(html_string, 'html.parser')
    table = soup.find('table')
    
    if not table:
        return html_string
    
    rows = table.find_all('tr')
    row_counter = 1
    
    for row in rows:
        # Check if this row already has <th> tags (it's a header row)
        if row.find('th'):
            continue
        
        # This is a data row - add row index as first <th> cell
        row_th = soup.new_tag('th')
        row_th.string = f"row {row_counter}"
        row.insert(0, row_th)
        
        row_counter += 1
    
    return table.prettify()


####################################################################################################