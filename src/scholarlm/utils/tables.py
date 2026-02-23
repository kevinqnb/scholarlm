import pandas as pd
from bs4 import BeautifulSoup


def table_extract(
    table_df: pd.DataFrame,
    row_indices: list[str],
    column_indices: list[str],
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
    flattened_table = table_df.reset_index(drop=True).T.reset_index().T.reset_index(drop=True)
    row_mask = ((flattened_table == val).sum(axis=1) for i, val in enumerate(row_indices))
    row_mask = pd.concat(row_mask, axis=1).all(axis=1)
    row_idx = row_mask[row_mask].index[0]
    column_mask = ((flattened_table == val).sum(axis=0) for i, val in enumerate(column_indices))
    column_mask = pd.concat(column_mask, axis=1).all(axis=1)
    column_idx = column_mask[column_mask].index[0]

    return flattened_table.iat[row_idx, column_idx]


def add_row_names(html_string: str) -> str:
    """
    Add row names (indices) to all non-header rows in an HTML table.

    Adds a <th> tag at the beginning of each row that doesn't already
    contain <th> tags, numbering them sequentially starting from 1.

    Args:
        html_string: String containing HTML table

    Returns:
        String containing table with row names added
    """
    soup = BeautifulSoup(html_string, "html.parser")
    table = soup.find("table")

    if not table:
        return html_string

    rows = table.find_all("tr")
    row_counter = 1

    for row in rows:
        if row.find("th"):
            continue
        row_th = soup.new_tag("th")
        row_th.string = f"row {row_counter}"
        row.insert(0, row_th)
        row_counter += 1

    return table.prettify()
