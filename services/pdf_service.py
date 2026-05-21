"""
PDF Service — text extraction per-page and full-doc.
Uses pdfplumber for text-based PDFs.
Uses PyMuPDF for rendering PDF pages as PNG images and text search.
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
    Returns (png_bytes, 0.0, 0.0, 1.0, 1.0) — full page fractions.
    Used as fallback when text search fails (for visual/symbol elements).
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
    Finds the EXACT pixel position of text on a PDF page using PyMuPDF.
    Returns normalized bbox {top, left, width, height} (0.0–1.0).

    Searches only within the label content zone (default 35%–92% of page height):
      - Skips top 35%  → avoids the Redline annotation header
                          (numbered change instructions in red at top of page)
      - Skips bottom 8% → avoids the footer disclaimer text
                           (Work Instruction ref, WI-0244, Rev AV, Page 1 of 1)

    This ensures:
      Change 1 — "Rx Only" finds the symbol on the label, not the header text
      Change 2 — "REV. A" finds the revision field, not the footer "Rev AV"
      Change 3 — "METAGLENE" finds the descriptor line in the label body

    Works for any company's label — no hardcoded assumptions, just geometry.
    Falls back to {"found": False} if nothing found in the content zone.
    """
    try:
        import fitz
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        if page_num >= len(doc):
            page_num = 0
        page = doc[page_num]
        page_w = page.rect.width
        page_h = page.rect.height

        # Content zone boundaries in page points
        zone_top    = page_h * search_from_top
        zone_bottom = page_h * search_from_bottom

        def in_zone(rects):
            """Keep only matches inside the label content zone."""
            return [r for r in rects if r.y0 > zone_top and r.y1 < zone_bottom]

        # ── Try 1: exact text match ────────────────────────────────────
        instances = in_zone(page.search_for(search_text, quads=False))

        # ── Try 2: first 3 words ───────────────────────────────────────
        if not instances:
            short = " ".join(search_text.split()[:3])
            if short != search_text:
                instances = in_zone(page.search_for(short, quads=False))

        # ── Try 3: first 2 words ───────────────────────────────────────
        if not instances:
            short2 = " ".join(search_text.split()[:2])
            if short2 != search_text:
                instances = in_zone(page.search_for(short2, quads=False))

        if not instances:
            print(f"[TEXT SEARCH] '{search_text}' not found in zone "
                  f"{search_from_top:.0%}–{search_from_bottom:.0%}")
            return {"found": False, "bbox": None, "method": "text_search"}

        # Use the first match — add small padding for visibility
        r = instances[0]
        pad_x = page_w * 0.005
        pad_y = page_h * 0.003

        result = {
            "found": True,
            "method": "text_search",
            "bbox": {
                "top":    max(0.0, (r.y0 - pad_y) / page_h),
                "left":   max(0.0, (r.x0 - pad_x) / page_w),
                "width":  min(1.0, (r.x1 - r.x0 + pad_x * 2) / page_w),
                "height": min(1.0, (r.y1 - r.y0 + pad_y * 2) / page_h),
            },
        }
        print(f"[TEXT SEARCH] ✅ '{search_text}' → top={result['bbox']['top']:.3f} "
              f"left={result['bbox']['left']:.3f}")
        return result

    except Exception as e:
        logger.error(f"Text search failed for '{search_text}': {e}")
        return {"found": False, "bbox": None, "method": "text_search"}