"""
PDF Service — text extraction per-page and full-doc.
Uses pdfplumber for text-based PDFs.
Uses PyMuPDF for rendering PDF pages as PNG images.
Uses pytesseract OCR for pixel-perfect text location on any PDF type.
"""

import io
import logging

logger = logging.getLogger(__name__)


def extract_text_from_pdf(file_bytes: bytes) -> tuple[str, int]:
    """Returns (full_text, page_count) — all pages joined."""
    pages = extract_pages_text(file_bytes)
    return "\n\n".join(pages), len(pages)


def extract_pages_text(file_bytes: bytes) -> list[str]:
    """
    Returns list of per-page text strings.
    Critical for redline PDFs where page 1 = label 1, page 2 = label 2.
    """
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            pages = []
            for page in pdf.pages:
                t = page.extract_text() or ""
                pages.append(t)
            if any(p.strip() for p in pages):
                return pages
    except ImportError:
        logger.warning("pdfplumber not installed — run: pip install pdfplumber")
    except Exception as e:
        logger.warning(f"pdfplumber failed: {e}")

    try:
        from pdf2image import convert_from_bytes
        import pytesseract
        images = convert_from_bytes(file_bytes)
        return [pytesseract.image_to_string(img) for img in images]
    except ImportError:
        logger.warning("pdf2image/pytesseract not installed — returning empty pages")
    except Exception as e:
        logger.error(f"OCR fallback failed: {e}")

    return [""]


def render_page_as_png(file_bytes: bytes, page_num: int = 0) -> bytes:
    """
    Render one PDF page to PNG bytes at 200 DPI.
    page_num is 0-based (page 0 = first page).
    """
    try:
        import fitz
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        if page_num >= len(doc):
            page_num = 0
        page = doc[page_num]
        mat = fitz.Matrix(200 / 72, 200 / 72)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        return pix.tobytes("png")
    except ImportError:
        logger.error("PyMuPDF not installed — run: pip install PyMuPDF")
        raise
    except Exception as e:
        logger.error(f"PDF render failed: {e}")
        raise


def render_label_area_as_png(
    file_bytes: bytes, page_num: int = 0
) -> tuple[bytes, float, float, float, float]:
    """
    Renders the full page at 200 DPI.
    Returns (png_bytes, 0.0, 0.0, 1.0, 1.0).
    Used as input for gpt-4o vision fallback.
    """
    try:
        png = render_page_as_png(file_bytes, page_num)
        return png, 0.0, 0.0, 1.0, 1.0
    except Exception as e:
        logger.error(f"render_label_area_as_png failed: {e}")
        raise


def find_text_bbox_on_page(
    file_bytes: bytes,
    page_num: int,
    search_text: str,
    search_from_top: float = 0.35,
    search_from_bottom: float = 0.92,
) -> dict:
    """
    PyMuPDF text search — fast but only works on text-based PDFs.
    Returns {"found": False} for vector/image PDFs like DePuy labels.
    """
    try:
        import fitz
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        if page_num >= len(doc):
            page_num = 0
        page = doc[page_num]
        page_w = page.rect.width
        page_h = page.rect.height

        zone_top    = page_h * search_from_top
        zone_bottom = page_h * search_from_bottom

        def in_zone(rects):
            return [r for r in rects if r.y0 > zone_top and r.y1 < zone_bottom]

        instances = in_zone(page.search_for(search_text, quads=False))

        if not instances:
            short = " ".join(search_text.split()[:3])
            if short != search_text:
                instances = in_zone(page.search_for(short, quads=False))

        if not instances:
            return {"found": False, "bbox": None, "method": "text_search"}

        r = instances[0]
        pad_x = page_w * 0.005
        pad_y = page_h * 0.003

        return {
            "found": True,
            "method": "text_search",
            "bbox": {
                "top":    max(0.0, (r.y0 - pad_y) / page_h),
                "left":   max(0.0, (r.x0 - pad_x) / page_w),
                "width":  min(1.0, (r.x1 - r.x0 + pad_x * 2) / page_w),
                "height": min(1.0, (r.y1 - r.y0 + pad_y * 2) / page_h),
            },
        }
    except Exception as e:
        logger.error(f"Text search failed: {e}")
        return {"found": False, "bbox": None, "method": "text_search"}


def ocr_find_text_bbox(
    png_bytes: bytes,
    search_terms: list[str],
    page_height_fraction_min: float = 0.30,
    page_height_fraction_max: float = 0.92,
) -> dict:
    """
    OCR-based pixel-perfect text location.
    Works on vector PDFs, image PDFs, scanned PDFs — any format.

    Key insight from DePuy label debug:
    - METAGLENE found at top%=0.554 — single word search works perfectly
    - REV.B found at top%=0.640 on Final Label — no space between REV. and B
    - REV on Redline has handwritten slash — not readable, vision handles it
    - Rx Only is a graphic symbol — not readable, vision handles it
    """
    try:
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = \
            r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        from PIL import Image

        img = Image.open(io.BytesIO(png_bytes))
        img_w, img_h = img.size

        zone_min_px = img_h * page_height_fraction_min
        zone_max_px = img_h * page_height_fraction_max

        data = pytesseract.image_to_data(
            img,
            output_type=pytesseract.Output.DICT,
            config="--psm 11"
        )

        n = len(data["text"])

        for term in search_terms:
            term_words = term.lower().split()
            term_len   = len(term_words)

            for i in range(n - term_len + 1):
                window = [
                    (data["text"][i + j] or "").lower().strip()
                    for j in range(term_len)
                ]

                if window != term_words:
                    continue

                confidences = []
                for j in range(term_len):
                    c = str(data["conf"][i + j])
                    if c.lstrip("-").isdigit():
                        confidences.append(int(c))

                if not confidences or max(confidences) < 40:
                    continue

                x0_px = min(data["left"][i + j]                          for j in range(term_len))
                y0_px = min(data["top"][i + j]                           for j in range(term_len))
                x1_px = max(data["left"][i + j] + data["width"][i + j]   for j in range(term_len))
                y1_px = max(data["top"][i + j]  + data["height"][i + j]  for j in range(term_len))

                if y0_px < zone_min_px or y1_px > zone_max_px:
                    print(f"[OCR] '{term}' found at y={y0_px:.0f}px "
                          f"(top={y0_px/img_h:.3f}) outside zone "
                          f"({zone_min_px:.0f}–{zone_max_px:.0f}px) — skipping")
                    continue

                pad_x = img_w * 0.005
                pad_y = img_h * 0.003

                result = {
                    "found": True,
                    "method": "ocr",
                    "bbox": {
                        "top":    max(0.0, (y0_px - pad_y) / img_h),
                        "left":   max(0.0, (x0_px - pad_x) / img_w),
                        "width":  min(1.0, (x1_px - x0_px + pad_x * 2) / img_w),
                        "height": min(1.0, (y1_px - y0_px + pad_y * 2) / img_h),
                    }
                }
                print(f"[OCR] ✅ '{term}' → "
                      f"top={result['bbox']['top']:.3f} "
                      f"left={result['bbox']['left']:.3f} "
                      f"w={result['bbox']['width']:.3f} "
                      f"h={result['bbox']['height']:.3f}")
                return result

        print(f"[OCR] nothing found for: {search_terms}")
        return {"found": False, "bbox": None, "method": "ocr"}

    except ImportError:
        logger.error(
            "pytesseract not installed — run: pip install pytesseract pillow"
        )
        return {"found": False, "bbox": None, "method": "ocr"}
    except Exception as e:
        logger.error(f"OCR search failed: {e}")
        return {"found": False, "bbox": None, "method": "ocr"}