from bs4 import BeautifulSoup


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
