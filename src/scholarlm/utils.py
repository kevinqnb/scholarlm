import os
import json
import base64
from io import BytesIO
import numpy as np
import pandas as pd
import networkx as nx
from rapidfuzz import fuzz
import torch
from PIL import Image
import pytesseract
from bs4 import BeautifulSoup
import subprocess
from pypdf import PdfReader
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Type, Union


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


def _try_coerce_value(value: Any, target_type: Type[Any]) -> Any:
    """Best-effort coercion of a single value to `target_type`.

    - If coercion fails, returns the original `value`.
    - Only performs very light cleaning intended for PDF-extracted numerics.
    """
    if value is None:
        return value

    # If it's already the right type, keep it.
    if isinstance(value, target_type):
        return value

    try:
        if target_type is str:
            return str(value)

        if target_type in (int, float):
            # Handle common cases like "1,234" or leading/trailing whitespace
            if isinstance(value, str):
                s = value.strip().replace(",", "")
                # also handle unicode minus
                s = s.replace("−", "-")
                if target_type is int:
                    # Allow "12.0"-style strings by going through float first
                    return int(float(s))
                return float(s)

            # Non-string numeric-like input
            if target_type is int:
                return int(value)
            return float(value)

        # Generic callable conversion
        return target_type(value)
    except Exception:
        return value


####################################################################################################


def load_and_process_results(
    json_path: str,
    *,
    unit_conversion_table: Mapping[str, Mapping[str, Any]],
    attribute_types: Optional[Mapping[Any, Type[Any]]] = None,
    drop_keys: Optional[Iterable[str]] = None,
    drop_attrs: Optional[Iterable[Any]] = None,
    attribute_col: str = "attribute",
    value_col: str = "value",
    unit_col: str = "units",
    out_col: str = "processed_value",
) -> pd.DataFrame:
    """Load experiment results JSON, deduplicate, optionally coerce value types, and apply unit conversions.

    New behavior vs your original helper:
    - `attribute_types` can specify an expected Python type per attribute (e.g. {"tn": float}).
      For each row we will *try* to coerce `value_col` to that type (after light cleaning like
      removing commas). If coercion fails, the original value is preserved.

    Parameters
    ----------
    json_path:
        Path to a JSON file containing a list of dict records.
    unit_conversion_table:
        Mapping like: unit_conversion_table[attribute][unit] -> multiplicative factor.
        If an attribute is string-valued in your data, use an empty dict (or omit the attribute)
        so no conversion is applied.
    attribute_types:
        Optional mapping: attribute_types[attribute] -> Python type (e.g., int/float/str).
        This function does not enforce—only attempts coercion.
    drop_keys:
        Optional iterable of keys to drop from each record before creating the DataFrame.
    drop_attrs:
        Optional iterable of attribute values to drop (i.e., drop rows where `attribute_col` is in this set).
    attribute_col:
        Name of the column containing the attribute names (used for type coercion and unit conversion).
    value_col:
        Name of the column containing the values to be processed.
    unit_col:
        Name of the column containing the unit names (used for unit conversion).
    out_col:
        Name of the new column to store processed values after type coercion and unit conversion.

    Returns
    -------
    A processed DataFrame with an additional `out_col` column.
    """
    with open(json_path, "r") as f:
        records: List[Dict[str, Any]] = json.load(f)

    if drop_keys:
        drop_set = set(drop_keys)
        records = [{k: v for k, v in r.items() if k not in drop_set} for r in records]
    if drop_attrs:
        drop_attr_set = set(drop_attrs)
        records = [r for r in records if r.get(attribute_col) not in drop_attr_set]


    df = pd.DataFrame(records)
    df = df.dropna(subset=[value_col])

    # Optional per-attribute type coercion
    if attribute_types and attribute_col in df.columns and value_col in df.columns:
        coerced_values: List[Any] = []
        for _, row in df.iterrows():
            attr = row.get(attribute_col)
            val = row.get(value_col)
            target_type = attribute_types.get(attr)
            if target_type is None:
                coerced_values.append(val)
            else:
                coerced_values.append(_try_coerce_value(val, target_type))
        df[value_col] = coerced_values

    # Unit conversion: only apply when (attribute, unit) exists in the table.
    processed: List[Any] = []
    for _, row in df.iterrows():
        attr = row.get(attribute_col)
        unit = row.get(unit_col)
        val = row.get(value_col)

        attr_table = unit_conversion_table.get(attr, {}) if attr is not None else {}
        factor = attr_table.get(unit) if isinstance(attr_table, Mapping) else None

        if factor is None:
            processed.append(val)
        else:
            try:
                processed.append(val * factor)
            except Exception:
                # If the user provided a conversion for non-numeric data, do not crash.
                processed.append(val)

    df[out_col] = processed
    df = df.dropna(subset=[out_col])
    df = df.reset_index(drop=True)
    return df


####################################################################################################


import math
from typing import Dict, Iterable, List, Optional, Tuple, Union


def match_datasets(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    *,
    strict_matching: Dict[str, str],
    fuzzy_matching: Optional[Dict[str, str]] = None,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], List[float]]:
    """Match rows across two dataframes.

    The procedure builds candidate edges between rows that satisfy *strict* criteria and
    optionally scores them with a fuzzy similarity. It can then compute a maximum-weight
    bipartite matching (1-1 alignment) using NetworkX.

    Parameters
    ----------
    df_left, df_right:
        Dataframes to match.

    strict_matching:
        Mapping from column name in `df_left` -> column name in `df_right` that must be
        strictly equal. If both values are numeric they are compared with `np.isclose`.
        For object/string values, comparison is exact (after optional normalization).

    fuzzy_matching:
        Mapping from column name in `df_left` -> column name in `df_right` that are
        compared with fuzzy ratios and averaged to produce an edge weight in [0, 1].

    Returns
    -------
    (matching, edges, edge_weights)
        matching: list of (left_index, right_index) pairs.
        edges: list of (left_index, right_index) candidate edges.
        edge_weights: list of edge weights aligned with `edges`.

    Notes
    -----
    - This function expects `df_left` and `df_right` to have stable integer positions.
      It uses positional indices from `DataFrame.iterrows()` (which yield the index value).
      For typical usage, call `reset_index(drop=True)` beforehand.
    """
    float_atol = 1e-3
    float_rtol = 0.0

    if fuzzy_matching is None:
        fuzzy_matching = {}

    if not isinstance(strict_matching, dict) or len(strict_matching) == 0:
        raise ValueError("strict_matching must be a non-empty dict mapping left_col -> right_col")

    missing_left = [c for c in strict_matching.keys() if c not in df_left.columns]
    missing_right = [c for c in strict_matching.values() if c not in df_right.columns]
    if missing_left:
        raise KeyError(f"Columns missing from df_left: {missing_left}")
    if missing_right:
        raise KeyError(f"Columns missing from df_right: {missing_right}")

    missing_left_f = [c for c in fuzzy_matching.keys() if c not in df_left.columns]
    missing_right_f = [c for c in fuzzy_matching.values() if c not in df_right.columns]
    if missing_left_f:
        raise KeyError(f"Columns missing from df_left (fuzzy): {missing_left_f}")
    if missing_right_f:
        raise KeyError(f"Columns missing from df_right (fuzzy): {missing_right_f}")

    def _is_null(x) -> bool:
        # pd.isna handles None/np.nan in a vectorized-safe way
        return bool(pd.isna(x))

    def _normalize_obj(x):
        if _is_null(x):
            return None
        if isinstance(x, str):
            return x.lower().strip()
        return x

    def _is_numeric_scalar(x) -> bool:
        # bool is a subclass of int; exclude it
        if isinstance(x, (bool, np.bool_)):
            return False
        return isinstance(x, (int, float, np.integer, np.floating)) and not _is_null(x)

    def _strict_equal(v_left, v_right) -> bool:
        # Null semantics: treat null != null to be conservative
        if _is_null(v_left) or _is_null(v_right):
            return False
        
        # Handle numeric equivalence if both sides look numeric
        if _is_numeric_scalar(v_left) and _is_numeric_scalar(v_right):
            return bool(np.isclose(float(v_left), float(v_right), atol=float_atol, rtol=float_rtol))

        # Otherwise, exact equality after normalization
        if _normalize_obj(v_left) == _normalize_obj(v_right):
            #print(f"  Strict match: {v_left!r} <-> {v_right!r}")
            return True
        else:
            #print(f"  Strict mismatch: {v_left!r} <-> {v_right!r}")
            return False

    def _fuzzy_score(v_left, v_right) -> Optional[float]:
        if _is_null(v_left) or _is_null(v_right):
            return None

        s_left = _normalize_obj(v_left)
        s_right = _normalize_obj(v_right)

        # If not strings, fall back to exact equality -> score 1/0
        if not isinstance(s_left, str) or not isinstance(s_right, str):
            return 1.0 if s_left == s_right else 0.0

        return float(fuzz.ratio(s_left, s_right)) / 100.0

    edges: List[Tuple[int, int]] = []
    edge_weights: List[float] = []

    strict_items = list(strict_matching.items())
    fuzzy_items = list(fuzzy_matching.items())

    # Naive O(N*M) scan; if this becomes a bottleneck, we can pre-index df_right on strict keys.
    for i, row_l in df_left.iterrows():
        for j, row_r in df_right.iterrows():
            # 1) Strict gate
            strict_ok = True
            for k, (c_l, c_r) in enumerate(strict_items):
                v_l, v_r = row_l[c_l], row_r[c_r]
                if not _strict_equal(v_l, v_r):
                    strict_ok = False
                    break
            if not strict_ok:
                continue

            # 2) Fuzzy scoring
            if len(fuzzy_items) == 0:
                score = 1.0
            else:
                scores: List[float] = []
                for c_l, c_r in fuzzy_items:
                    s = _fuzzy_score(row_l[c_l], row_r[c_r])
                    if s is None:
                        continue
                    scores.append(s)

                if len(scores) == 0:
                    continue

                score = float(np.mean(scores))

            edges.append((int(i), int(j)))
            edge_weights.append(float(score))

    # 3) Maximum weight matching on a bipartite graph
    if len(edges) == 0:
        return [], edges, edge_weights

    G = nx.Graph()
    G.add_edges_from(
        [(f"L_{i}", f"R_{j}", {"weight": w}) for (i, j), w in zip(edges, edge_weights)]
    )

    matching_nodes = nx.algorithms.matching.max_weight_matching(G, maxcardinality=False)

    matching: List[Tuple[int, int]] = []
    for u, v in matching_nodes:
        if u.startswith("L_"):
            li = int(u[2:])
            rj = int(v[2:])
        else:
            li = int(v[2:])
            rj = int(u[2:])
        matching.append((li, rj))

    # Stable ordering can be helpful for downstream inspection
    matching.sort(key=lambda x: (x[0], x[1]))

    return matching, edges, edge_weights


####################################################################################################


def matching_precision_recall(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    *,
    strict_matching: Dict[str, str],
    fuzzy_matching: Optional[Dict[str, str]] = None,
    **match_kwargs,
) -> Tuple[float, float]:
    """Estimate recall/precision under the matching rules.

    Recall is computed relative to `df_left`, precision relative to `df_right`.
    """

    total_left = int(df_left.shape[0])
    total_right = int(df_right.shape[0])

    matching, _edges, _weights = match_datasets(
        df_left,
        df_right,
        strict_matching=strict_matching,
        fuzzy_matching=fuzzy_matching,
        **match_kwargs,
    )

    tp = len(matching)
    precision = tp / total_right if total_right > 0 else 0.0
    recall = tp / total_left if total_left > 0 else 0.0

    return recall, precision


####################################################################################################