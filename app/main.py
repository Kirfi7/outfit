import os
import json
import base64
import re
from io import BytesIO
from typing import Optional, Any, Dict

import requests
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

# --- image normalize (HEIC -> JPEG, resize) ---
from PIL import Image

# pillow-heif registers HEIF opener for PIL
try:
    import pillow_heif  # type: ignore
    pillow_heif.register_heif_opener()
except Exception:
    pillow_heif = None


AITUNNEL_URL = "https://api.aitunnel.ru/v1/chat/completions"
AITUNNEL_KEY = os.getenv("AITUNNEL_KEY")  # НЕ хардкодь ключ в коде
MODEL_NAME = os.getenv("AITUNNEL_MODEL", "gemini-2.5-flash-lite")

if not AITUNNEL_KEY:
    # чтобы не “молчать” и не падать странно
    raise RuntimeError("AITUNNEL_KEY env var is not set")

CLOTHES_PROMPT = """
Ты — ассистент по оценке размеров одежды по фото человека.
Твоя задача: по двум фото (1: спереди, 2: сзади) оценить:
1) окружность головы в см (head_cm)
2) размер футболки/верха (tshirt.ru: российский размер 40–62 шаг 2; tshirt.intl: XS,S,M,L,XL,XXL,3XL)
3) размер брюк/низа (pants.ru: российский размер 40–62 шаг 2; pants.waist_in: талия в дюймах 24–40, если можешь)

КРИТИЧНО:
- Верни ТОЛЬКО валидный JSON, без пояснений, без markdown, без кода-блоков.
- Всегда возвращай все ключи: head_cm, tshirt{ru,intl}, pants{ru,waist_in}, confidence, notes
- Если не можешь определить поле — поставь null.
- confidence: число 0.0..1.0
- notes: <=120 символов, что мешает оценке.

Фото:
- Image 1: front
- Image 2: back
"""

SHOES_PROMPT = """
Ты — ассистент по оценке размера обуви по фото сбоку.
По одному фото (вид сбоку) оцени shoe_eu (EU размер 34–48).

КРИТИЧНО:
- Верни ТОЛЬКО валидный JSON, без пояснений, без markdown.
- Всегда возвращай ключи: shoe_eu, confidence, notes
- confidence: 0.0..1.0
- notes: <=120 символов.

Фото:
- Image 1: side
"""

app = FastAPI()


class TshirtOut(BaseModel):
    ru: Optional[int]
    intl: Optional[str]


class PantsOut(BaseModel):
    ru: Optional[int]
    waist_in: Optional[int]


class ClothesOut(BaseModel):
    head_cm: Optional[int]
    tshirt: TshirtOut
    pants: PantsOut
    confidence: float
    notes: str


class ShoesOut(BaseModel):
    shoe_eu: Optional[int]
    confidence: float
    notes: str


def _normalize_to_jpeg(image_bytes: bytes, max_side: int = 1280, quality: int = 82) -> bytes:
    """
    Любой вход (heic/jpg/png/webp) -> JPEG bytes + resize.
    """
    try:
        img = Image.open(BytesIO(image_bytes))
    except Exception as e:
        # Если прилетел HEIC, а pillow-heif не установлен/не завёлся — будет тут.
        hint = ""
        if pillow_heif is None:
            hint = " (HEIC? install pillow-heif)"
        raise HTTPException(400, f"Cannot decode image{hint}: {e}")

    # исправляем ориентацию (iPhone часто пишет её в EXIF)
    try:
        from PIL import ImageOps
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass

    # приводим к RGB (JPEG требует RGB)
    if img.mode not in ("RGB",):
        img = img.convert("RGB")

    # resize
    w, h = img.size
    m = max(w, h)
    if m > max_side:
        scale = max_side / float(m)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    out = BytesIO()
    img.save(out, format="JPEG", quality=quality, optimize=True)
    return out.getvalue()


def _to_data_url_jpeg(jpeg_bytes: bytes) -> str:
    b64 = base64.b64encode(jpeg_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def _extract_json(text: str) -> Dict[str, Any]:
    if not text:
        raise ValueError("Empty model response")

    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        raise ValueError("No JSON object found in response")
    return json.loads(m.group(0))


def _call_aitunnel(messages: Any) -> Dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {AITUNNEL_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0,
        "top_p": 1,
        "max_tokens": 600,
    }

    try:
        r = requests.post(AITUNNEL_URL, headers=headers, json=payload, timeout=120)
    except Exception as e:
        raise HTTPException(502, f"aitunnel request failed: {e}")

    if r.status_code != 200:
        # вернём больше, чтобы понять причину (но не бесконечно)
        body = (r.text or "")[:1500]
        raise HTTPException(502, f"aitunnel error {r.status_code}: {body}")

    data = r.json()
    try:
        text = data["choices"][0]["message"]["content"]
    except Exception:
        raise HTTPException(502, f"unexpected response: {data}")

    try:
        return _extract_json(text)
    except Exception as e:
        raise HTTPException(502, f"bad model json: {e}; raw={str(text)[:500]}")


def _safe_int(x):
    try:
        if x is None:
            return None
        # иногда приходит "52.0"
        s = str(x).strip().replace(".0", "")
        return int(s)
    except Exception:
        return None


@app.post("/sizes/clothes", response_model=ClothesOut)
async def sizes_clothes(
    front: UploadFile = File(...),
    back: UploadFile = File(...),
):
    front_bytes = await front.read()
    back_bytes = await back.read()
    if not front_bytes or not back_bytes:
        raise HTTPException(400, "front/back images required")

    # ВАЖНО: нормализуем в JPEG
    front_jpeg = _normalize_to_jpeg(front_bytes)
    back_jpeg = _normalize_to_jpeg(back_bytes)

    front_url = _to_data_url_jpeg(front_jpeg)
    back_url = _to_data_url_jpeg(back_jpeg)

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": CLOTHES_PROMPT},
            {"type": "image_url", "image_url": {"url": front_url}},
            {"type": "image_url", "image_url": {"url": back_url}},
        ],
    }]

    j = _call_aitunnel(messages)

    tshirt = j.get("tshirt") or {}
    pants = j.get("pants") or {}

    out = {
        "head_cm": _safe_int(j.get("head_cm")),
        "tshirt": {
            "ru": _safe_int(tshirt.get("ru")),
            "intl": (tshirt.get("intl") or None),
        },
        "pants": {
            "ru": _safe_int(pants.get("ru")),
            "waist_in": _safe_int(pants.get("waist_in")),
        },
        "confidence": float(j.get("confidence") or 0.0),
        "notes": str(j.get("notes") or ""),
    }

    return ClothesOut(**out)


@app.post("/sizes/shoes", response_model=ShoesOut)
async def sizes_shoes(
    side: UploadFile = File(...),
):
    side_bytes = await side.read()
    if not side_bytes:
        raise HTTPException(400, "side image required")

    side_jpeg = _normalize_to_jpeg(side_bytes)
    side_url = _to_data_url_jpeg(side_jpeg)

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": SHOES_PROMPT},
            {"type": "image_url", "image_url": {"url": side_url}},
        ],
    }]

    j = _call_aitunnel(messages)

    out = {
        "shoe_eu": _safe_int(j.get("shoe_eu")),
        "confidence": float(j.get("confidence") or 0.0),
        "notes": str(j.get("notes") or ""),
    }
    return ShoesOut(**out)