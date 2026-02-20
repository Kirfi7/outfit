from typing import Any, Dict

import httpx
from fastapi import HTTPException

from app.core.config import get_settings
from app.services.parsing import extract_json


async def aitunnel_chat(messages: Any) -> Dict[str, Any]:
    settings = get_settings()
    headers = {
        "Authorization": f"Bearer {settings.aitunnel_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": settings.aitunnel_model,
        "messages": messages,
        "temperature": 0,
        "top_p": 1,
        "max_tokens": 600,
    }

    timeout = httpx.Timeout(connect=20.0, read=120.0, write=60.0, pool=60.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(settings.aitunnel_url, headers=headers, json=payload)

    if r.status_code != 200:
        raise HTTPException(502, f"aitunnel error {r.status_code}: {(r.text or '')[:1500]}")

    data = r.json()
    try:
        text = data["choices"][0]["message"]["content"]
    except Exception:
        raise HTTPException(502, f"unexpected aitunnel response: {str(data)[:1500]}")

    try:
        return extract_json(text)
    except Exception as e:
        raise HTTPException(502, f"bad model json: {e}; raw={(str(text)[:500])}")
