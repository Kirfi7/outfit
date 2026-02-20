import base64
from typing import Any, Dict, List, Tuple

import httpx
from fastapi import HTTPException

from ..core.config import get_settings
from ..core.http import make_httpx_client


def img_part(jpeg_bytes: bytes) -> Dict[str, Any]:
    b64 = base64.b64encode(jpeg_bytes).decode("utf-8")
    return {"inlineData": {"mimeType": "image/jpeg", "data": b64}}


async def gemini_generate_image(prompt: str, labeled_images: List[Tuple[str, bytes]]) -> Tuple[bytes, str]:
    settings = get_settings()
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{settings.gemini_model}:generateContent?key={settings.gemini_api_key}"
    )

    parts: List[Dict[str, Any]] = [{"text": prompt}]
    for label, jpeg in labeled_images:
        parts.append({"text": f"{label}:"})
        parts.append(img_part(jpeg))

    payload = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {"imageConfig": {"aspectRatio": settings.image_ar, "imageSize": settings.image_size}},
    }

    timeout = httpx.Timeout(connect=30.0, read=600.0, write=60.0, pool=60.0)
    async with make_httpx_client(timeout, settings.proxy_url) as client:
        r = await client.post(url, json=payload, headers={"Content-Type": "application/json"})

    if r.status_code < 200 or r.status_code >= 300:
        raise HTTPException(502, f"gemini {r.status_code}: {(r.text or '')[:1500]}")

    data = r.json()

    for cand in (data.get("candidates") or []):
        content = (cand or {}).get("content") or {}
        for part in (content.get("parts") or []):
            inline = (part or {}).get("inlineData")
            if inline and inline.get("data"):
                mime = inline.get("mimeType") or "image/png"
                return base64.b64decode(inline["data"]), mime

    raise HTTPException(502, f"gemini returned no image: {str(data)[:2000]}")
