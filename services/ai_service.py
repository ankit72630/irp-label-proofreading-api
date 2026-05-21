"""
AI Service — OpenAI only.

Two OpenAI APIs used:
  1. gpt-4o-mini  → text comparison (change verification, fast + cheap)
  2. gpt-4o       → vision (highlight bounding boxes — fallback for symbols/logos only)

.env file in backend/:
  OPENAI_API_KEY=sk-...
"""

import base64
import json
import logging
import os
from typing import Optional
import httpx

logger = logging.getLogger(__name__)

OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
TEXT_MODEL   = "gpt-4o-mini"
VISION_MODEL = "gpt-4o"

VERIFY_PROMPT = """You are an expert pharmaceutical label compliance analyst.

You will be given:
1. A CHANGE INSTRUCTION from a redline document
2. REDLINE TEXT (original label with change markup)
3. FINAL LABEL TEXT (the supposedly updated label)

Your task: Determine if the change instruction was correctly applied in the final label.

Respond ONLY with valid JSON — no extra text, no markdown fences:
{{
  "outcome": "pass",
  "confidence": 0.9,
  "explanation": "one clear sentence here",
  "group": "Regulatory"
}}

Use one of these exact outcome values: pass, fail, warn
Use one of these exact group values: Regulatory, Formatting, Content, PLM, Logo, General

Change Instruction: {instruction}

Redline Text:
{redline_text}

Final Label Text:
{final_text}
"""

LOGO_PROMPT = """You are analyzing a pharmaceutical label image for visual compliance.

Tasks:
1. Detect all logos, symbols, and brand marks visible on this label
2. Check if any Rx-only symbol is present
3. Note any visual marks that might need regulatory review

Return ONLY valid JSON:
{{
  "logos_found": [
    {{
      "type": "rx_symbol",
      "description": "what it is",
      "bbox": {{"top": 0.5, "left": 0.1, "width": 0.05, "height": 0.03}},
      "compliance_note": "any concern or OK"
    }}
  ],
  "rx_symbol_present": false,
  "total_visual_elements": 3,
  "overall_note": "one sentence summary"
}}
"""


class AIService:
    def __init__(self):
        self._api_key: Optional[str] = os.getenv("OPENAI_API_KEY")

    async def initialize(self):
        if self._api_key:
            logger.info(f"✅ OpenAI API ready — text={TEXT_MODEL}, vision={VISION_MODEL}")
        else:
            logger.error("❌ OPENAI_API_KEY not set — add it to backend/.env")

    async def shutdown(self):
        pass

    async def verify_change(
        self, instruction: str, redline_text: str, final_text: str
    ) -> dict:
        """TEXT: did this change get applied? Uses gpt-4o-mini."""
        if not self._api_key:
            return self._no_key_result()
        prompt = VERIFY_PROMPT.format(
            instruction=instruction,
            redline_text=redline_text[:1500],
            final_text=final_text[:1500],
        )
        result = await self._chat(
            model=TEXT_MODEL,
            system=(
                "You are a pharmaceutical label compliance analyst. "
                "Always respond with valid JSON only. "
                "Never include markdown fences or extra text. "
                "The outcome field must be exactly one of: pass, fail, warn"
            ),
            user=prompt,
        )
        return result or self._fallback_result()

    async def detect_highlight_bbox(
        self,
        page_image_bytes: bytes,
        instruction: str,
        side: str = "redline",
        hint: dict = None,
    ) -> dict:
        """
        VISION fallback — only called when PyMuPDF text search fails.
        Used for graphical elements: Rx symbol, logos, barcodes, CE marks.

        hint: optional dict {top, left, width, height} from the Redline bbox,
              used to tell gpt-4o approximately where to look on the Final Label.
        """
        if not self._api_key:
            return {"found": False, "bbox": None, "description": "no API key"}

        b64 = base64.b64encode(page_image_bytes).decode()

        # Build location hint text for final label
        hint_text = ""
        if hint and side == "final":
            hint_text = (
                f"\n\nLOCATION HINT: On the Redline version of this label, "
                f"this element was found at approximately "
                f"{hint['top'] * 100:.0f}% from the top and "
                f"{hint['left'] * 100:.0f}% from the left. "
                f"Look in this same region on this Final Label."
            )

        if side == "redline":
            prompt = f"""You are analyzing a pharmaceutical label PDF page image.
This is the REDLINE document — original label with change annotations.

Find the region relating to this change instruction: "{instruction}"

This is a VISUAL element (symbol, logo, graphic) — text search already failed to find it.
Look for:
- The Rx Only symbol (℞ or Rx graphic near the bottom symbols row)
- Any red numbered annotation markers (1. 2. 3.) pointing to the element
- Logos, barcodes, CE marks, NON STERILE triangle

Return ONLY valid JSON with normalized coordinates (0.0 to 1.0):
{{
  "found": true,
  "bbox": {{"top": 0.72, "left": 0.55, "width": 0.06, "height": 0.04}},
  "description": "what you found and where"
}}

If not found: {{"found": false, "bbox": null, "description": "not found"}}"""

        else:
            prompt = f"""You are analyzing a pharmaceutical label PDF page image.
This is the FINAL LABEL — updated version after changes were applied.

For this change instruction: "{instruction}"{hint_text}

This is a VISUAL element — find where it was or where it should be.
Look for the symbols row (warning triangle ⚠, IFU booklet, Rx area).
Even if the element was removed, find the surrounding symbols area.

Return ONLY valid JSON with normalized coordinates (0.0 to 1.0):
{{
  "found": true,
  "bbox": {{"top": 0.72, "left": 0.55, "width": 0.06, "height": 0.04}},
  "description": "what you found or the surrounding area"
}}

If completely unable to locate: {{"found": false, "bbox": null, "description": "not found"}}"""

        result = await self._vision_chat(
            model=VISION_MODEL,
            system=(
                "You are a precise pharmaceutical label document layout analyst. "
                "Always respond with valid JSON only. "
                "Never include markdown fences or extra text."
            ),
            text_prompt=prompt,
            image_b64=b64,
        )
        return result or {"found": False, "bbox": None, "description": "detection failed"}

    async def detect_logos_and_symbols(self, page_image_bytes: bytes) -> dict:
        """VISION: detect logos, Rx symbols, barcodes on a label page. Uses gpt-4o."""
        if not self._api_key:
            return {"logos_found": [], "rx_symbol_present": False, "overall_note": "no API key"}
        b64 = base64.b64encode(page_image_bytes).decode()
        result = await self._vision_chat(
            model=VISION_MODEL,
            system=(
                "You are a pharmaceutical label visual compliance analyst. "
                "Always respond with valid JSON only. Never include markdown fences."
            ),
            text_prompt=LOGO_PROMPT,
            image_b64=b64,
        )
        return result or {
            "logos_found": [],
            "rx_symbol_present": False,
            "overall_note": "detection failed",
        }

    async def _chat(self, model: str, system: str, user: str) -> Optional[dict]:
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                r = await client.post(
                    OPENAI_API_URL,
                    headers=self._headers(),
                    json={
                        "model": model,
                        "response_format": {"type": "json_object"},
                        "temperature": 0.1,
                        "max_tokens": 400,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user",   "content": user},
                        ],
                    },
                )
                r.raise_for_status()
                return self._parse(r.json()["choices"][0]["message"]["content"])
        except Exception as e:
            logger.warning(f"OpenAI text call failed: {e}")
            return None

    async def _vision_chat(
        self, model: str, system: str, text_prompt: str, image_b64: str
    ) -> Optional[dict]:
        try:
            async with httpx.AsyncClient(timeout=90.0) as client:
                r = await client.post(
                    OPENAI_API_URL,
                    headers=self._headers(),
                    json={
                        "model": model,
                        "response_format": {"type": "json_object"},
                        "temperature": 0.1,
                        "max_tokens": 600,
                        "messages": [
                            {"role": "system", "content": system},
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/png;base64,{image_b64}",
                                            "detail": "high",
                                        },
                                    },
                                    {"type": "text", "text": text_prompt},
                                ],
                            },
                        ],
                    },
                )
                r.raise_for_status()
                return self._parse(r.json()["choices"][0]["message"]["content"])
        except Exception as e:
            logger.warning(f"OpenAI vision call failed: {e}")
            return None

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def _parse(self, raw: str) -> Optional[dict]:
        try:
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                cleaned = "\n".join(lines).strip()
            data = json.loads(cleaned)
            if "outcome" in data:
                if data["outcome"] not in ("pass", "fail", "warn"):
                    data["outcome"] = "warn"
            return data
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse failed: {e} | raw={raw[:300]}")
            return None
        except Exception as e:
            logger.warning(f"Parse error: {e} | raw={raw[:300]}")
            return None

    def _no_key_result(self) -> dict:
        return {
            "outcome": "warn",
            "confidence": 0.0,
            "explanation": "OPENAI_API_KEY not set in .env",
            "group": "General",
        }

    def _fallback_result(self) -> dict:
        return {
            "outcome": "warn",
            "confidence": 0.5,
            "explanation": "AI analysis unavailable — manual review required.",
            "group": "General",
        }

    @property
    def backend_name(self) -> str:
        return (
            f"OpenAI ({TEXT_MODEL} + {VISION_MODEL} vision)"
            if self._api_key
            else "None — set OPENAI_API_KEY"
        )