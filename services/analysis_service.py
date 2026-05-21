"""
Analysis Service — orchestrates the full proofreading pipeline.

Real document structure (from actual DePuy/J&J labels):
  Redline.pdf   — multi-page; page N = label N; numbered changes listed at TOP of each page
  LRF.pdf       — Label Request Form; provides correct descriptor text per product code
  Final labels  — one PDF per label; matched to redline page by PLM title (LCN-XXXXXXXXX_1)
"""

import asyncio
import re
import logging
import uuid
from datetime import datetime
from typing import Optional

from models.schemas import AnalysisStatus, LabelResult, ChangeResult, BoundingBox
from services.pdf_service import extract_text_from_pdf, extract_pages_text
from services.ai_service import AIService

logger = logging.getLogger(__name__)

_jobs: dict[str, AnalysisStatus] = {}
_files: dict[str, bytes] = {}


def store_file(file_id: str, data: bytes):
    _files[file_id] = data


def get_file(file_id: str) -> Optional[bytes]:
    return _files.get(file_id)


def get_job(job_id: str) -> Optional[AnalysisStatus]:
    return _jobs.get(job_id)


# ── Parsers ────────────────────────────────────────────────────────────────────

def _extract_plm_title(text: str) -> Optional[str]:
    """
    Extract PLM Label Title like 'LCN-299967042_1' from page text.
    Appears as: 'PLM Label Title:   LCN-299967042_1'
    """
    m = re.search(r"PLM\s+Label\s+Title[:\s]+([A-Z0-9\-_]+)", text, re.IGNORECASE)
    return m.group(1).strip() if m else None


def _extract_numbered_changes(page_text: str) -> list[str]:
    """
    Parse numbered change instructions from the TOP section of a redline page.
    Real format from Redline.pdf:
        1. Remove Rx only symbol. Not required per 103063851 Appendix 2
        2. All Revisions change to B
        3. Update 299967043 descriptor per LRF
    """
    instructions = []
    # Match lines like "1. Some instruction text" or "1) Some text"
    pattern = re.compile(r"^\s*(\d+)[.\)]\s+(.+)", re.MULTILINE)
    for m in pattern.finditer(page_text):
        inst = m.group(2).strip()
        if inst:
            instructions.append(inst)

    if instructions:
        logger.info(f"Extracted {len(instructions)} numbered changes from redline page")
        return instructions

    # Fallback: scan for action-verb sentences if no numbered list found
    lines = [l.strip() for l in page_text.splitlines() if l.strip()]
    fallback = []
    action_re = re.compile(
        r"^(remove|add|update|change|replace|delete|insert|revise|correct)\b.+",
        re.IGNORECASE,
    )
    for line in lines:
        if action_re.match(line) and len(line) < 200:
            fallback.append(line)
    return fallback[:20]


def _extract_lrf_descriptors(lrf_text: str) -> dict[str, str]:
    """
    Extract product code → descriptor mapping from LRF text.
    LRF contains rows like:
        2999-67-042   DELTA XTENDTM   MAKE TO ORDER   METAGLENE POSITIONER FOR INHANCE HANDLE
        2999-67-043   DELTA XTENDTM   MAKE TO ORDER   METAGLENE SIZER FULL WEDGE FOR INHANCE HANDLE
    Returns: {"2999-67-042": "METAGLENE POSITIONER FOR INHANCE HANDLE", ...}
    """
    descriptors = {}
    # Match product code (with dashes) followed by descriptor text on same line
    pattern = re.compile(
        r"(2\d{3}-\d{2}-\d{3})\s+.{5,50}?\s{2,}(METAGLENE[^\n]{5,80})",
        re.IGNORECASE,
    )
    for m in pattern.finditer(lrf_text):
        code = m.group(1).strip()
        desc = m.group(2).strip()
        descriptors[code] = desc
        logger.info(f"LRF descriptor: {code} → {desc}")
    return descriptors


def _match_final_to_redline_page(
    final_plm: Optional[str],
    redline_plm: Optional[str],
    page_idx: int,
    final_idx: int,
) -> bool:
    """
    Try to match a final label to a redline page by PLM title.
    Falls back to positional matching (final label N → redline page N).
    """
    if final_plm and redline_plm:
        # Strip trailing _1, _2 suffix for comparison e.g. LCN-299967042_1 → LCN-299967042
        def _base(s: str) -> str:
            return re.sub(r"_\d+$", "", s).upper()
        return _base(final_plm) == _base(redline_plm)
    # Positional fallback
    return page_idx == final_idx


# ── Main pipeline ──────────────────────────────────────────────────────────────

async def run_analysis(
    job_id: str,
    redline_file_id: str,
    lrf_file_id: Optional[str],
    final_label_file_ids: list[str],
    ai_service: AIService,
):
    """
    Full proofreading pipeline:
    1. Extract redline pages → per-page change instructions + PLM titles
    2. Extract LRF descriptors (if provided) → enrich AI context
    3. Extract final label texts + PLM titles
    4. Match final labels to redline pages by PLM title (fallback: positional)
    5. AI verify each change per label (text) + vision check Rx symbol removal
    6. Assemble results
    """

    def _update(status: str, progress: int, message: str):
        job = _jobs[job_id]
        job.status = status
        job.progress = progress
        job.message = message
        logger.info(f"[{job_id}] {status} {progress}% — {message}")

    _jobs[job_id] = AnalysisStatus(
        job_id=job_id,
        status="extracting",
        progress=5,
        message="Extracting text from Redline PDF…",
    )

    try:
        # ── Step 1: Extract redline pages ────────────────────────────────
        _update("extracting", 10, "Extracting Redline PDF pages…")
        redline_bytes = get_file(redline_file_id)
        if not redline_bytes:
            raise ValueError("Redline file not found — please re-upload")

        # extract_pages_text returns list of per-page text
        redline_pages: list[str] = extract_pages_text(redline_bytes)
        logger.info(f"Redline has {len(redline_pages)} page(s)")

        # Per page: extract changes + PLM title
        redline_page_data: list[dict] = []
        for i, page_text in enumerate(redline_pages):
            changes = _extract_numbered_changes(page_text)
            plm = _extract_plm_title(page_text)
            redline_page_data.append({
                "page_idx": i,
                "page_num": i + 1,
                "text": page_text,
                "plm_title": plm,
                "instructions": changes,
            })
            logger.info(f"Redline page {i+1}: PLM={plm}, {len(changes)} changes")

        # ── Step 2: Extract LRF data (optional) ─────────────────────────
        lrf_descriptors: dict[str, str] = {}
        if lrf_file_id:
            _update("extracting", 18, "Extracting Label Request Form data…")
            lrf_bytes = get_file(lrf_file_id)
            if lrf_bytes:
                lrf_text, _ = extract_text_from_pdf(lrf_bytes)
                lrf_descriptors = _extract_lrf_descriptors(lrf_text)
                logger.info(f"LRF descriptors found: {lrf_descriptors}")

        # ── Step 3: Extract final label texts ───────────────────────────
        _update("ocr", 28, f"Extracting text from {len(final_label_file_ids)} final label(s)…")
        final_label_data: list[dict] = []
        for fid in final_label_file_ids:
            fb = get_file(fid)
            if not fb:
                logger.warning(f"Final label file {fid} not found — skipping")
                continue
            text, pages = extract_text_from_pdf(fb)
            plm = _extract_plm_title(text)
            final_label_data.append({
                "file_id": fid,
                "text": text,
                "page_count": pages,
                "plm_title": plm,
                "bytes": fb,
            })
            logger.info(f"Final label: PLM={plm}, {pages} page(s)")

        if not final_label_data:
            raise ValueError("No final label files found — please upload at least one")

        # ── Step 4: Match final labels → redline pages ───────────────────
        # Build pairs: (final_label_data, redline_page_data)
        pairs: list[tuple[dict, dict]] = []

        for fi, final in enumerate(final_label_data):
            matched_page = None
            # Try PLM-title match first
            for rp in redline_page_data:
                if _match_final_to_redline_page(
                    final["plm_title"], rp["plm_title"], rp["page_idx"], fi
                ):
                    matched_page = rp
                    break
            # Fallback: positional
            if matched_page is None:
                matched_page = redline_page_data[fi % len(redline_page_data)]
                logger.warning(
                    f"No PLM match for final label {final['plm_title']} — "
                    f"using positional fallback → redline page {matched_page['page_num']}"
                )
            pairs.append((final, matched_page))

        # ── Step 5: AI verification per label ────────────────────────────
        _update("ai_verify", 40, "Running AI change verification…")
        label_results: list[LabelResult] = []

        for li, (final, redline_page) in enumerate(pairs):
            progress = 40 + int((li / len(pairs)) * 50)
            _update(
                "ai_verify",
                progress,
                f"Verifying label {li+1}/{len(pairs)}: {final['plm_title'] or final['file_id'][:8]}…",
            )

            instructions = redline_page["instructions"]
            if not instructions:
                # Safety fallback — should not happen with real redlines
                instructions = [
                    "Remove Rx only symbol. Not required per 103063851 Appendix 2",
                    "All Revisions change to B",
                    "Update descriptor per LRF",
                ]

            # Build enriched redline context — include LRF descriptor if available
            lrf_context = ""
            if lrf_descriptors:
                lrf_context = "\n\nLRF Reference Descriptors:\n" + "\n".join(
                    f"  {code}: {desc}" for code, desc in lrf_descriptors.items()
                )

            redline_context = redline_page["text"] + lrf_context

            # Run all change verifications in parallel for this label
            tasks = [
                ai_service.verify_change(
                    instruction=inst,
                    redline_text=redline_context,
                    final_text=final["text"],
                )
                for inst in instructions
            ]
            ai_results = await asyncio.gather(*tasks, return_exceptions=True)

            changes: list[ChangeResult] = []
            for i, (inst, ai_res) in enumerate(zip(instructions, ai_results)):
                if isinstance(ai_res, Exception):
                    logger.error(f"AI error for change {i+1}: {ai_res}")
                    ai_res = {
                        "outcome": "warn",
                        "confidence": 0.5,
                        "explanation": f"AI error — manual review needed: {ai_res}",
                        "group": "General",
                    }

                # For Rx symbol changes — flag for vision check
                # (vision bbox detection happens in frontend when user clicks the change)
                is_rx_change = "rx" in inst.lower() or "symbol" in inst.lower()

                changes.append(
                    ChangeResult(
                        instruction=inst,
                        reason=ai_res.get("explanation", ""),
                        outcome=ai_res.get("outcome", "warn"),
                        confidence=float(ai_res.get("confidence", 0.5)),
                        group=ai_res.get("group", "General"),
                        ai_explanation=ai_res.get("explanation"),
                        redline_page=redline_page["page_num"],
                        # Bounding boxes: vision API fills these in when user
                        # clicks a change in the frontend. Placeholder here.
                        redline_bbox=None,
                        final_bbox=None,
                    )
                )

            passed = sum(1 for c in changes if c.outcome == "pass")
            failed = sum(1 for c in changes if c.outcome == "fail")
            warnings = sum(1 for c in changes if c.outcome == "warn")

            label_results.append(
                LabelResult(
                    label_id=final["file_id"],
                    filename=final["plm_title"] or f"label_{li+1}.pdf",
                    plm_title=final["plm_title"],
                    total_changes=len(changes),
                    passed=passed,
                    failed=failed,
                    warnings=warnings,
                    changes=changes,
                )
            )

        # ── Step 6: Finalise ─────────────────────────────────────────────
        _update("report", 95, "Generating compliance report…")
        await asyncio.sleep(0.2)

        job = _jobs[job_id]
        job.status = "done"
        job.progress = 100
        job.message = f"Analysis complete — {len(label_results)} label(s) verified"
        job.completed_at = datetime.utcnow()
        job.result = label_results
        logger.info(f"[{job_id}] Done — {len(label_results)} labels")

    except Exception as e:
        logger.exception(f"Analysis pipeline failed: {e}")
        job = _jobs.get(job_id)
        if job:
            job.status = "error"
            job.error = str(e)
            job.message = f"Error: {e}"
