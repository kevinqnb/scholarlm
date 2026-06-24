import base64
import subprocess
from io import BytesIO

import pytesseract
from PIL import Image
from pypdf import PdfReader


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
        timeout=30,
    )

    if result.returncode != 0:
        raise ValueError(f"pdfinfo failed:  {result.stderr}")

    for line in result.stdout.splitlines():
        if "MediaBox" in line:
            parts = line.split(":", 1)[1].strip().split()
            if len(parts) >= 4:
                x0, y0, x1, y1 = map(float, parts[:4])
                return x1 - x0, y1 - y0

    raise ValueError("MediaBox not found in PDF info")


def load_pdf_page(
    pdf_path: str,
    page_num: int,
    target_longest_dim: int = 2048,
) -> Image.Image:
    """
    Render a PDF page to a high-quality PIL Image.

    Args:
        pdf_path: Path to PDF file
        page_num: Page number (1-indexed)
        target_longest_dim: Target size for longest dimension in pixels

    Returns:
        PIL Image object
    """
    width, height = get_pdf_page_dimensions(pdf_path, page_num)
    dpi = int(target_longest_dim * 72 / max(width, height))

    result = subprocess.run(
        ["pdftoppm", "-png", "-f", str(page_num), "-l", str(page_num), "-r", str(dpi), pdf_path],
        capture_output=True,
        timeout=120,
    )

    if result.returncode != 0:
        raise RuntimeError(f"pdftoppm failed: {result.stderr.decode()}")

    image = Image.open(BytesIO(result.stdout))
    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


def correct_image_orientation(pil_image: Image.Image) -> Image.Image:
    """
    Detects orientation of a PIL image using Tesseract OSD and returns a rotated image corrected to upright.

    Args:
        pil_image (PIL.Image.Image): Input image

    Returns:
        PIL.Image.Image: Upright-corrected image
    """
    try:
        osd_output = pytesseract.image_to_osd(pil_image)
        rotate_angle = 0
        for line in osd_output.splitlines():
            if "Rotate" in line:
                rotate_angle = int(line.split(":")[-1].strip())
                break
    except pytesseract.TesseractError as e:
        print("Tesseract OSD failed, proceeding without orientation correction.")
        print(e)
        print()
        rotate_angle = 0

    if rotate_angle == 0:
        return pil_image
    return pil_image.rotate(-rotate_angle, expand=True)


def encode_pil_image(pil_image: Image.Image) -> str:
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


def process_pdf(
    pdf_path: str,
    target_longest_dim: int = 2048,
    correct_orientation: bool = True,
) -> list[str]:
    """
    Process all pages of a PDF and return base64-encoded images, one per page.

    Args:
        pdf_path: Path to PDF file
        target_longest_dim: Target size for longest dimension
        correct_orientation: Run Tesseract OSD to detect and correct page rotation.
            Disable for faster processing when pages are reliably upright.

    Returns:
        List of base64-encoded strings for each page
    """
    reader = PdfReader(pdf_path)
    num_pages = len(reader.pages)

    results = []
    for page_num in range(1, num_pages + 1):
        pil_image = load_pdf_page(pdf_path, page_num, target_longest_dim)
        if correct_orientation:
            pil_image = correct_image_orientation(pil_image)
        results.append(encode_pil_image(pil_image))

    return results
