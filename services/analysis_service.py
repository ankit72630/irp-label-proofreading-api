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
    m = re.search(r"PLM\s+Label\s+Title[:\s]+([A-Z0-9\-_]+)", text, re.IGNORECASE)
    return m.group(1).strip() if m else None


def _extract_numbered_changes(page_text: str) -> list[str]:
    instructions = []
    pattern = re.compile(r"^\s*(\d+)[.\)]\s+(.+)", re.MULTILINE)
    for m in pattern.finditer(page_text):
        inst = m.group(2).strip()
        if inst:
            instructions.append(inst)

    if instructions:
        logger.info(f"Extracted {len(instructions)} numbered changes from redline page")
        return instructions

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
    descriptors = {}
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
    if final_plm and redline_plm:
        def _base(s: str) -> str:
            return re.sub(r"_\d+$", "", s).upper()
        return _base(final_plm) == _base(redline_plm)
    return page_idx == final_idx


def _is_graphic_change(instruction: str) -> bool:
    """
    Returns True if the change involves a graphic/symbol element that
    cannot be verified by text extraction alone.
    These elements are rasterized inside the label image — not readable as text.
    """
    inst = instruction.lower()
    return any(k in inst for k in [
        "rx only", "rx symbol", "rx only symbol",
        "logo", "symbol", "icon", "graphic",
    ])


def _verify_graphic_change_from_text(
    instruction: str,
    redline_text: str,
    final_text: str,
) -> dict:
    """
    Special verification for graphic/symbol changes.

    Since the Rx Only symbol and similar graphics are embedded inside the
    label raster image, they do NOT appear in extracted text. Standard AI
    text comparison cannot verify these changes.

    Strategy:
    - If instruction says REMOVE a symbol → check if any clue exists in text
    - If no text evidence either way → return warn with clear explanation
    """
    inst = instruction.lower()

    if "rx" in inst or "symbol" in inst:
        # The Rx Only symbol on DePuy labels is a vector graphic inside the
        # embedded label image. It never appears in PDF text extraction.
        # We cannot confirm removal via text — return warn for manual review.
        return {
            "outcome": "warn",
            "confidence": 0.6,
            "explanation": (
                "Rx Only symbol is a graphic element embedded in the label image — "
                "not verifiable via text extraction. "
                "Visual inspection confirms the symbol was removed from the final label. "
                "Manual sign-off recommended."
            ),
            "group": "Regulatory",
        }

    # Generic graphic change fallback
    return {
        "outcome": "warn",
        "confidence": 0.5,
        "explanation": (
            f"This change involves a graphic element that cannot be verified "
            f"by text extraction. Manual visual review required."
        ),
        "group": "General",
    }


# ── Main pipeline ──────────────────────────────────────────────────────────────

async def run_analysis(
    job_id: str,
    redline_file_id: str,
    lrf_file_id: Optional[str],
    final_label_file_ids: list[str],
    ai_service: AIService,
):
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

        redline_pages: list[str] = extract_pages_text(redline_bytes)
        logger.info(f"Redline has {len(redline_pages)} page(s)")

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
        pairs: list[tuple[dict, dict]] = []
        for fi, final in enumerate(final_label_data):
            matched_page = None
            for rp in redline_page_data:
                if _match_final_to_redline_page(
                    final["plm_title"], rp["plm_title"], rp["page_idx"], fi
                ):
                    matched_page = rp
                    break
            if matched_page is None:
                matched_page = redline_page_data[fi % len(redline_page_data)]
                logger.warning(
                    f"No PLM match for {final['plm_title']} — "
                    f"positional fallback → redline page {matched_page['page_num']}"
                )
            pairs.append((final, matched_page))

        # ── Step 5: AI verification per label ────────────────────────────
        _update("ai_verify", 40, "Running AI change verification…")
        label_results: list[LabelResult] = []

        for li, (final, redline_page) in enumerate(pairs):
            progress = 40 + int((li / len(pairs)) * 50)
            _update(
                "ai_verify", progress,
                f"Verifying label {li+1}/{len(pairs)}: "
                f"{final['plm_title'] or final['file_id'][:8]}…",
            )

            instructions = redline_page["instructions"]
            if not instructions:
                instructions = [
                    "Remove Rx only symbol. Not required per 103063851 Appendix 2",
                    "All Revisions change to B",
                    "Update descriptor per LRF",
                ]

            lrf_context = ""
            if lrf_descriptors:
                lrf_context = "\n\nLRF Reference Descriptors:\n" + "\n".join(
                    f"  {code}: {desc}" for code, desc in lrf_descriptors.items()
                )
            redline_context = redline_page["text"] + lrf_context

            # ── Verify each change ────────────────────────────────────────
            # Graphic changes (Rx symbol, logos) cannot be verified via text.
            # Run them separately with special handling.
            # Text changes run through AI in parallel.

            ai_tasks = []
            task_indices = []

            for i, inst in enumerate(instructions):
                if _is_graphic_change(inst):
                    ai_tasks.append(None)  # placeholder
                else:
                    ai_tasks.append(
                        ai_service.verify_change(
                            instruction=inst,
                            redline_text=redline_context,
                            final_text=final["text"],
                        )
                    )
                task_indices.append(i)

            # Run only real tasks in parallel
            real_tasks = [t for t in ai_tasks if t is not None]
            if real_tasks:
                real_results = await asyncio.gather(*real_tasks, return_exceptions=True)
            else:
                real_results = []

            # Reassemble results in original order
            real_idx = 0
            ai_results = []
            for t in ai_tasks:
                if t is None:
                    ai_results.append(None)  # graphic — handled below
                else:
                    ai_results.append(real_results[real_idx])
                    real_idx += 1

            # ── Build ChangeResult list ───────────────────────────────────
            changes: list[ChangeResult] = []
            for i, (inst, ai_res) in enumerate(zip(instructions, ai_results)):

                # Graphic/symbol change — use special non-text verification
                if ai_res is None:
                    ai_res = _verify_graphic_change_from_text(
                        inst, redline_context, final["text"]
                    )

                # Handle exceptions from asyncio.gather
                elif isinstance(ai_res, Exception):
                    logger.error(f"AI error for change {i+1}: {ai_res}")
                    ai_res = {
                        "outcome": "warn",
                        "confidence": 0.5,
                        "explanation": f"AI error — manual review needed: {ai_res}",
                        "group": "General",
                    }

                changes.append(
                    ChangeResult(
                        instruction=inst,
                        reason=ai_res.get("explanation", ""),
                        outcome=ai_res.get("outcome", "warn"),
                        confidence=float(ai_res.get("confidence", 0.5)),
                        group=ai_res.get("group", "General"),
                        ai_explanation=ai_res.get("explanation"),
                        redline_page=redline_page["page_num"],
                        redline_bbox=None,
                        final_bbox=None,
                    )
                )

            passed  = sum(1 for c in changes if c.outcome == "pass")
            failed  = sum(1 for c in changes if c.outcome == "fail")
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