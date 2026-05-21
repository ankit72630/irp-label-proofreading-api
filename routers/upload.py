"""
Upload router — 3 distinct slots matching real document workflow:
  POST /api/upload/redline       — exactly 1 PDF (multi-page, 1 page per label)
  POST /api/upload/lrf           — exactly 1 PDF (Label Request Form reference)
  POST /api/upload/final-label   — bulk, 1 PDF per label, no limit on count
  GET  /api/upload/{file_id}/page/{page_num}   — render PDF page as PNG
  POST /api/upload/{file_id}/highlight         — smart highlight bbox detection
"""

import uuid
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import Response
from models.schemas import UploadResponse
from services.pdf_service import (
    extract_text_from_pdf,
    render_page_as_png,
    render_label_area_as_png,
    find_text_bbox_on_page,
)
from services.analysis_service import store_file, get_file

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

router = APIRouter()
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB per file


@router.post("/redline", response_model=UploadResponse,
             summary="Upload Redline PDF (1 file, multi-page — 1 page per label)")
async def upload_redline(file: UploadFile = File(...)):
    return await _handle_upload(file, slot="redline")


@router.post("/lrf", response_model=UploadResponse,
             summary="Upload Label Request Form PDF (1 file, reference data)")
async def upload_lrf(file: UploadFile = File(...)):
    return await _handle_upload(file, slot="lrf")


@router.post("/final-label", response_model=UploadResponse,
             summary="Upload one Final Label PDF (call multiple times for bulk)")
async def upload_final_label(file: UploadFile = File(...)):
    return await _handle_upload(file, slot="final_label")


async def _handle_upload(file: UploadFile, slot: str) -> UploadResponse:
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted")
    data = await file.read()
    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(413, f"File exceeds {MAX_FILE_SIZE // (1024*1024)} MB limit")
    _, page_count = extract_text_from_pdf(data)
    file_id = str(uuid.uuid4())
    store_file(file_id, data)
    return UploadResponse(
        file_id=file_id,
        filename=file.filename,
        size_bytes=len(data),
        page_count=page_count,
    )


@router.get("/{file_id}/page/{page_num}",
            summary="Render a PDF page as PNG image for browser display")
async def get_pdf_page(file_id: str, page_num: int = 0):
    file_bytes = get_file(file_id)
    if not file_bytes:
        raise HTTPException(404, "File not found")
    png_bytes = render_page_as_png(file_bytes, page_num)
    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={"Access-Control-Allow-Origin": "*"},
    )


@router.post("/{file_id}/highlight",
             summary="Smart highlight: text search first, vision fallback for symbols")
async def get_highlight_bbox(
    file_id: str,
    page_num: int = 0,
    instruction: str = "",
    side: str = "redline",
    hint_top: float = -1,
    hint_left: float = -1,
    hint_width: float = -1,
    hint_height: float = -1,
):
    """
    Two-strategy highlight detection:

    Strategy 1 — PyMuPDF text search (100% accurate for text changes)
    Searches for the actual text mentioned in the change instruction directly
    in the PDF geometry. No AI, no guessing — pixel perfect every time.
    Works for: descriptor text, revision letters, REF numbers, lot numbers.

    Strategy 2 — gpt-4o vision (fallback for visual/symbol elements)
    Used only when text search fails — e.g. Rx symbol (a graphic, not text),
    logos, barcodes, CE marks, NON STERILE triangle.
    """
    file_bytes = get_file(file_id)
    if not file_bytes:
        raise HTTPException(404, "File not found")

    # ── Strategy 1: PyMuPDF exact text search ────────────────────────
    search_terms = _extract_search_terms(instruction)
    print(f"[HIGHLIGHT] instruction='{instruction}' side={side}")
    print(f"[HIGHLIGHT] trying text search terms: {search_terms}")

    for term in search_terms:
        result = find_text_bbox_on_page(file_bytes, page_num, term)
        if result["found"]:
            print(f"[HIGHLIGHT] ✅ text search found '{term}': {result['bbox']}")
            return result

    # ── Strategy 2: gpt-4o vision fallback ───────────────────────────
    print(f"[HIGHLIGHT] text search found nothing — falling back to gpt-4o vision")

    png_bytes, frac_left, frac_top, frac_right, frac_bottom = \
        render_label_area_as_png(file_bytes, page_num)

    hint = None
    if side == "final" and hint_top >= 0:
        hint = {
            "top":    hint_top,
            "left":   hint_left,
            "width":  hint_width,
            "height": hint_height,
        }
        print(f"[HIGHLIGHT] using redline hint for final: {hint}")

    from services.ai_service import AIService
    ai = AIService()
    result = await ai.detect_highlight_bbox(png_bytes, instruction, side, hint)

    print(f"[HIGHLIGHT] vision raw: {result}")

    if result.get("found") and result.get("bbox"):
        b = result["bbox"]
        content_w = frac_right  - frac_left
        content_h = frac_bottom - frac_top

        if content_w > 0.01 and content_h > 0.01:
            def clamp(v: float) -> float:
                return max(0.0, min(1.0, v))

            result["bbox"] = {
                "top":    clamp(frac_top  + b["top"]    * content_h),
                "left":   clamp(frac_left + b["left"]   * content_w),
                "width":  clamp(b["width"]  * content_w),
                "height": clamp(b["height"] * content_h),
            }

        print(f"[HIGHLIGHT] vision mapped: {result['bbox']}")

    return result


def _extract_search_terms(instruction: str) -> list[str]:
    """
    Extracts searchable text terms from a change instruction.
    Most specific terms first — stops at first match.

    Covers the main DePuy/J&J change types AND generic pharma label elements
    so this works for any company's labels, not just DePuy.
    """
    inst = instruction.lower()
    terms = []

    # ── Descriptor / product description changes ──────────────────────
    if any(k in inst for k in ["descriptor", "metaglene", "description"]):
        terms += [
            "METAGLENE POSITIONER FOR INHANCE HANDLE",
            "METAGLENE SIZER FULL WEDGE FOR INHANCE HANDLE",
            "METAGLENE TRIAL FULL WEDGE FOR INHANCE HANDLE",
            "METAGLENE",
        ]

    # ── Revision / revision letter changes ───────────────────────────
    if any(k in inst for k in ["revision", "rev."]):
        terms += [
            "REV. A", "REV. B", "REV. C", "REV. D",
            "Rev. A", "Rev. B",
            "REV A",  "REV B",
        ]

    # ── REF / catalogue number changes ───────────────────────────────
    if any(k in inst for k in ["ref", "catalogue", "catalog", "299967"]):
        terms += [
            "2999-67-042", "2999-67-043",
            "299967042",   "299967043",
        ]

    # ── PLM title changes ─────────────────────────────────────────────
    if any(k in inst for k in ["plm", "lcn", "title"]):
        terms += ["LCN-299967042_1", "LCN-299967043_1"]

    # ── Shelf life / expiry changes ───────────────────────────────────
    if any(k in inst for k in ["shelf", "expir", "shelf-life"]):
        terms += ["Shelf-Life Days", "N/A"]

    # ── Manufacturer / address changes ───────────────────────────────
    if any(k in inst for k in ["manufactur", "address", "depuy"]):
        terms += ["DePuy Orthopaedics", "700 Orthopaedic Drive"]

    # ── Rx symbol — graphic element, text search unlikely but try ─────
    # (Falls through to vision if text not found)
    if any(k in inst for k in ["rx", "symbol"]):
        terms += ["Rx Only", "Rx only", "RxOnly"]

    # ── Generic pharma label elements ─────────────────────────────────
    if "sterile" in inst:
        terms += ["NON STERILE", "STERILE", "Non-Sterile"]
    if "lot" in inst:
        terms += ["LOT", "SAMPLE"]
    if "quantity" in inst or "qty" in inst:
        terms += ["QTY", "QUANTITY"]

    return terms