import os
import json
import base64
import re
from typing import Optional, Any, Dict
import requests
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

AITUNNEL_URL = "https://api.aitunnel.ru/v1/chat/completions"
AITUNNEL_KEY = os.getenv("AITUNNEL_KEY", "sk-aitunnel-PgYvLoenSwR20GlGVhN5gxkKwYsXdNBx")
MODEL_NAME = os.getenv("AITUNNEL_MODEL", "gemini-2.5-flash-lite")

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
- confidence: число 0.0..1.0 (насколько уверен в оценке).
- notes: очень коротко (<=120 символов), что мешает оценке (например: "свободная одежда/нет масштаба/поза").

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
- Если не можешь определить shoe_eu — поставь null.
- confidence: 0.0..1.0
- notes: <=120 символов (например: "ступня не видна/нет масштаба/перспектива").

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


def _to_data_url(image_bytes: bytes, mime: str = "image/jpeg") -> str:
  b64 = base64.b64encode(image_bytes).decode("utf-8")
  return f"data:{mime};base64,{b64}"


def _extract_json(text: str) -> Dict[str, Any]:
  """
  Модели иногда оборачивают JSON в ```...```.
  Достанем первый JSON-объект из строки.
  """
  if not text:
    raise ValueError("Empty model response")

  # уберём code fences
  text = text.strip()
  text = re.sub(r"^```(?:json)?\s*", "", text)
  text = re.sub(r"\s*```$", "", text)

  # попробуем напрямую
  try:
    return json.loads(text)
  except Exception:
    pass

  # вытащим первый {...}
  m = re.search(r"\{.*\}", text, flags=re.S)
  if not m:
    raise ValueError("No JSON object found in response")
  return json.loads(m.group(0))


def _call_aitunnel(messages):
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
  r = requests.post(AITUNNEL_URL, headers=headers, json=payload, timeout=120)
  if r.status_code != 200:
    raise HTTPException(502, f"aitunnel error {r.status_code}: {r.text[:300]}")
  data = r.json()
  try:
    text = data["choices"][0]["message"]["content"]
  except Exception:
    raise HTTPException(502, f"unexpected response: {data}")
  return _extract_json(text)


@app.post("/sizes/clothes", response_model=ClothesOut)
async def sizes_clothes(
  front: UploadFile = File(...),
  back: UploadFile = File(...),
):
  front_bytes = await front.read()
  back_bytes = await back.read()
  if not front_bytes or not back_bytes:
    raise HTTPException(400, "front/back images required")

  front_url = _to_data_url(front_bytes, front.content_type or "image/jpeg")
  back_url = _to_data_url(back_bytes, back.content_type or "image/jpeg")

  messages = [{
    "role": "user",
    "content": [
      {"type": "text", "text": CLOTHES_PROMPT},
      {"type": "image_url", "image_url": {"url": front_url}},
      {"type": "image_url", "image_url": {"url": back_url}},
    ],
  }]

  j = _call_aitunnel(messages)

  # минимальная нормализация, чтобы не падать на "L " и т.п.
  def _safe_int(x):
    try:
      return int(x)
    except Exception:
      return None

  head_cm = _safe_int(j.get("head_cm"))
  tshirt = j.get("tshirt") or {}
  pants = j.get("pants") or {}

  out = {
    "head_cm": head_cm,
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

  # Pydantic провалидирует, что структура как надо
  return ClothesOut(**out)


@app.post("/sizes/shoes", response_model=ShoesOut)
async def sizes_shoes(
  side: UploadFile = File(...),
):
  side_bytes = await side.read()
  if not side_bytes:
    raise HTTPException(400, "side image required")

  side_url = _to_data_url(side_bytes, side.content_type or "image/jpeg")

  messages = [{
    "role": "user",
    "content": [
      {"type": "text", "text": SHOES_PROMPT},
      {"type": "image_url", "image_url": {"url": side_url}},
    ],
  }]

  j = _call_aitunnel(messages)

  def _safe_int(x):
    try:
      return int(x)
    except Exception:
      return None

  out = {
    "shoe_eu": _safe_int(j.get("shoe_eu")),
    "confidence": float(j.get("confidence") or 0.0),
    "notes": str(j.get("notes") or ""),
  }
  return ShoesOut(**out)