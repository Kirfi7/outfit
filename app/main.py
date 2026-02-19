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

# # pillow-heif registers HEIF opener for PIL
# try:
#     import pillow_heif  # type: ignore
#     pillow_heif.register_heif_opener()
# except Exception:
#     pillow_heif = None
#
#
# AITUNNEL_URL = "https://api.aitunnel.ru/v1/chat/completions"
# AITUNNEL_KEY = os.getenv("AITUNNEL_KEY", "sk-aitunnel-PgYvLoenSwR20GlGVhN5gxkKwYsXdNBx")  # НЕ хардкодь ключ в коде
# MODEL_NAME = os.getenv("AITUNNEL_MODEL", "gemini-2.5-flash-lite")
#
# if not AITUNNEL_KEY:
#     # чтобы не “молчать” и не падать странно
#     raise RuntimeError("AITUNNEL_KEY env var is not set")
#
# CLOTHES_PROMPT = """
# Ты — ассистент по оценке размеров одежды по фото человека.
# Твоя задача: по двум фото (1: спереди, 2: сзади) оценить:
# 1) окружность головы в см (head_cm)
# 2) размер футболки/верха (tshirt.ru: российский размер 40–62 шаг 2; tshirt.intl: XS,S,M,L,XL,XXL,3XL)
# 3) размер брюк/низа (pants.ru: российский размер 40–62 шаг 2; pants.waist_in: талия в дюймах 24–40, если можешь)
#
# КРИТИЧНО:
# - Верни ТОЛЬКО валидный JSON, без пояснений, без markdown, без кода-блоков.
# - Всегда возвращай все ключи: head_cm, tshirt{ru,intl}, pants{ru,waist_in}, confidence, notes
# - Если не можешь определить поле — поставь null.
# - confidence: число 0.0..1.0
# - notes: <=120 символов, что мешает оценке.
#
# Фото:
# - Image 1: front
# - Image 2: back
# """
#
# SHOES_PROMPT = """
# Ты — ассистент по оценке размера обуви по фото сбоку.
# По одному фото (вид сбоку) оцени shoe_eu (EU размер 34–48).
#
# КРИТИЧНО:
# - Верни ТОЛЬКО валидный JSON, без пояснений, без markdown.
# - Всегда возвращай ключи: shoe_eu, confidence, notes
# - confidence: 0.0..1.0
# - notes: <=120 символов.
#
# Фото:
# - Image 1: side
# """

app = FastAPI()
#
#
# class TshirtOut(BaseModel):
#     ru: Optional[int]
#     intl: Optional[str]
#
#
# class PantsOut(BaseModel):
#     ru: Optional[int]
#     waist_in: Optional[int]
#
#
# class ClothesOut(BaseModel):
#     head_cm: Optional[int]
#     tshirt: TshirtOut
#     pants: PantsOut
#     confidence: float
#     notes: str
#
#
# class ShoesOut(BaseModel):
#     shoe_eu: Optional[int]
#     confidence: float
#     notes: str
#
#
# def _normalize_to_jpeg(image_bytes: bytes, max_side: int = 1280, quality: int = 82) -> bytes:
#     """
#     Любой вход (heic/jpg/png/webp) -> JPEG bytes + resize.
#     """
#     try:
#         img = Image.open(BytesIO(image_bytes))
#     except Exception as e:
#         # Если прилетел HEIC, а pillow-heif не установлен/не завёлся — будет тут.
#         hint = ""
#         if pillow_heif is None:
#             hint = " (HEIC? install pillow-heif)"
#         raise HTTPException(400, f"Cannot decode image{hint}: {e}")
#
#     # исправляем ориентацию (iPhone часто пишет её в EXIF)
#     try:
#         from PIL import ImageOps
#         img = ImageOps.exif_transpose(img)
#     except Exception:
#         pass
#
#     # приводим к RGB (JPEG требует RGB)
#     if img.mode not in ("RGB",):
#         img = img.convert("RGB")
#
#     # resize
#     w, h = img.size
#     m = max(w, h)
#     if m > max_side:
#         scale = max_side / float(m)
#         new_w = max(1, int(w * scale))
#         new_h = max(1, int(h * scale))
#         img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
#
#     out = BytesIO()
#     img.save(out, format="JPEG", quality=quality, optimize=True)
#     return out.getvalue()
#
#
# def _to_data_url_jpeg(jpeg_bytes: bytes) -> str:
#     b64 = base64.b64encode(jpeg_bytes).decode("utf-8")
#     return f"data:image/jpeg;base64,{b64}"
#
#
# def _extract_json(text: str) -> Dict[str, Any]:
#     if not text:
#         raise ValueError("Empty model response")
#
#     text = text.strip()
#     text = re.sub(r"^```(?:json)?\s*", "", text)
#     text = re.sub(r"\s*```$", "", text)
#
#     try:
#         return json.loads(text)
#     except Exception:
#         pass
#
#     m = re.search(r"\{.*\}", text, flags=re.S)
#     if not m:
#         raise ValueError("No JSON object found in response")
#     return json.loads(m.group(0))
#
#
# def _call_aitunnel(messages: Any) -> Dict[str, Any]:
#     headers = {
#         "Authorization": f"Bearer {AITUNNEL_KEY}",
#         "Content-Type": "application/json",
#     }
#     payload = {
#         "model": MODEL_NAME,
#         "messages": messages,
#         "temperature": 0,
#         "top_p": 1,
#         "max_tokens": 600,
#     }
#
#     try:
#         r = requests.post(AITUNNEL_URL, headers=headers, json=payload, timeout=120)
#     except Exception as e:
#         raise HTTPException(502, f"aitunnel request failed: {e}")
#
#     if r.status_code != 200:
#         # вернём больше, чтобы понять причину (но не бесконечно)
#         body = (r.text or "")[:1500]
#         raise HTTPException(502, f"aitunnel error {r.status_code}: {body}")
#
#     data = r.json()
#     try:
#         text = data["choices"][0]["message"]["content"]
#     except Exception:
#         raise HTTPException(502, f"unexpected response: {data}")
#
#     try:
#         return _extract_json(text)
#     except Exception as e:
#         raise HTTPException(502, f"bad model json: {e}; raw={str(text)[:500]}")
#
#
# def _safe_int(x):
#     try:
#         if x is None:
#             return None
#         # иногда приходит "52.0"
#         s = str(x).strip().replace(".0", "")
#         return int(s)
#     except Exception:
#         return None
#
#
# @app.post("/sizes/clothes", response_model=ClothesOut)
# async def sizes_clothes(
#     front: UploadFile = File(...),
#     back: UploadFile = File(...),
# ):
#     front_bytes = await front.read()
#     back_bytes = await back.read()
#     if not front_bytes or not back_bytes:
#         raise HTTPException(400, "front/back images required")
#
#     # ВАЖНО: нормализуем в JPEG
#     front_jpeg = _normalize_to_jpeg(front_bytes)
#     back_jpeg = _normalize_to_jpeg(back_bytes)
#
#     front_url = _to_data_url_jpeg(front_jpeg)
#     back_url = _to_data_url_jpeg(back_jpeg)
#
#     messages = [{
#         "role": "user",
#         "content": [
#             {"type": "text", "text": CLOTHES_PROMPT},
#             {"type": "image_url", "image_url": {"url": front_url}},
#             {"type": "image_url", "image_url": {"url": back_url}},
#         ],
#     }]
#
#     j = _call_aitunnel(messages)
#
#     tshirt = j.get("tshirt") or {}
#     pants = j.get("pants") or {}
#
#     out = {
#         "head_cm": _safe_int(j.get("head_cm")),
#         "tshirt": {
#             "ru": _safe_int(tshirt.get("ru")),
#             "intl": (tshirt.get("intl") or None),
#         },
#         "pants": {
#             "ru": _safe_int(pants.get("ru")),
#             "waist_in": _safe_int(pants.get("waist_in")),
#         },
#         "confidence": float(j.get("confidence") or 0.0),
#         "notes": str(j.get("notes") or ""),
#     }
#
#     print(out)
#
#     return ClothesOut(**out)
#
#
# @app.post("/sizes/shoes", response_model=ShoesOut)
# async def sizes_shoes(
#     side: UploadFile = File(...),
# ):
#     side_bytes = await side.read()
#     if not side_bytes:
#         raise HTTPException(400, "side image required")
#
#     side_jpeg = _normalize_to_jpeg(side_bytes)
#     side_url = _to_data_url_jpeg(side_jpeg)
#
#     messages = [{
#         "role": "user",
#         "content": [
#             {"type": "text", "text": SHOES_PROMPT},
#             {"type": "image_url", "image_url": {"url": side_url}},
#         ],
#     }]
#
#     j = _call_aitunnel(messages)
#
#     out = {
#         "shoe_eu": _safe_int(j.get("shoe_eu")),
#         "confidence": float(j.get("confidence") or 0.0),
#         "notes": str(j.get("notes") or ""),
#     }
#
#     print(out)
#
#     return ShoesOut(**out)



import base64
from io import BytesIO
from typing import Optional, List, Dict, Any, Tuple

import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
from PIL import Image

# HEIC support (optional)
try:
    import pillow_heif  # type: ignore
    pillow_heif.register_heif_opener()
except Exception:
    pillow_heif = None

app = FastAPI()

# ================== HARD-CODED CONFIG (for local test) ==================
GEMINI_API_KEY = "AIzaSyAq09NEzkyit2K-bsKPDjG1bLJSmv1GQOg"
GEMINI_MODEL = "gemini-2.5-flash-image"

# Прокси: обычно схема http:// даже если тип "HTTPS" (это про CONNECT-туннель)
# ВСТАВЬ СВОИ LOGIN/PASSWORD/IP/PORT
PROXY_URL = "http://sve2Zh:1W4P9D@138.219.173.226:8000"
# Например: "http://sve2Zh:1W4P9D@138.219.173.226:8000"
# =======================================================================

if not GEMINI_API_KEY or "PASTE_" in GEMINI_API_KEY:
    raise RuntimeError("Set GEMINI_API_KEY in code for local test")
if not PROXY_URL or "LOGIN" in PROXY_URL:
    # если хочешь тест без прокси — поставь PROXY_URL = ""
    pass


TRYON_OUTFIT_PROMPT = """
ЗАДАЧА: ФОТО-РЕДАКТИРОВАНИЕ (НЕ генерация новой сцены).
Нужно отредактировать базовое фото человека (PERSON_BASE), изменив ТОЛЬКО его одежду.

ПОРЯДОК ИЗОБРАЖЕНИЙ (КРИТИЧНО):
- PRINT_CROP: крупный план принта — ЭТАЛОН ПРИНТА. Должен совпасть 1:1.
- TOP_FULL: полное фото верха — цвет/крой/посадка верха.
- BOTTOM: низ
- SHOES: (если есть)
- OUTER: (если есть)
- PERSON_BASE: человек — БАЗОВОЕ ФОТО, его нельзя менять кроме одежды.

РАЗРЕШЕНО МЕНЯТЬ ТОЛЬКО:
- верх/низ/обувь/верхнюю одежду на человеке.

ЗАПРЕЩЕНО (СТРОГО):
- менять лицо/личность/пол/возраст/бороду/прическу
- менять телосложение/рост/пропорции/позу
- менять фон/свет/перспективу/ракурс
- менять кадрирование/масштаб
- добавлять других людей/предметы

КРИТИЧНО ПРО ПРИНТ:
- Принт на груди должен ТОЧНО соответствовать PRINT_CROP: форма/контуры/цвета/расположение.
- Не перерисовывай "по мотивам" и не подменяй текст/картинку.

ЕСЛИ чего-то нет:
- Если SHOES или OUTER не переданы — НЕ добавляй их.

ВЫХОД:
Одно фотореалистичное изображение: тот же человек/фон, но в предоставленных вещах.
""".strip()


def _open_image(image_bytes: bytes) -> Image.Image:
    try:
        img = Image.open(BytesIO(image_bytes))
    except Exception as e:
        hint = " (HEIC? install pillow-heif)" if pillow_heif is None else ""
        raise HTTPException(400, f"Cannot decode image{hint}: {e}")

    try:
        from PIL import ImageOps
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass

    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def _resize_max(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    m = max(w, h)
    if m <= max_side:
        return img
    scale = max_side / float(m)
    return img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)


def _encode_jpeg(img: Image.Image, quality: int) -> bytes:
    out = BytesIO()
    img.save(out, format="JPEG", quality=quality, optimize=True)
    return out.getvalue()


def _make_center_crop(img: Image.Image, crop_w: float = 0.55, crop_h: float = 0.55) -> Image.Image:
    w, h = img.size
    cw = int(w * crop_w)
    ch = int(h * crop_h)
    left = max(0, (w - cw) // 2)
    top = max(0, (h - ch) // 2)
    return img.crop((left, top, left + cw, top + ch))


def _img_part(jpeg_bytes: bytes) -> Dict[str, Any]:
    b64 = base64.b64encode(jpeg_bytes).decode("utf-8")
    return {"inlineData": {"mimeType": "image/jpeg", "data": b64}}


def _make_client(timeout: httpx.Timeout) -> httpx.AsyncClient:
    """
    Создаём httpx клиент с прокси (если задан).
    На новых версиях httpx используется proxy=..., на старых — proxies=...
    """
    if not PROXY_URL:
        return httpx.AsyncClient(timeout=timeout, trust_env=False)

    try:
        # httpx >= 0.27
        return httpx.AsyncClient(timeout=timeout, trust_env=False, proxy=PROXY_URL)
    except TypeError:
        # старые версии
        return httpx.AsyncClient(timeout=timeout, trust_env=False, proxies={"http": PROXY_URL, "https": PROXY_URL})


async def _gemini_generate_image(
    *,
    model: str,
    prompt: str,
    labeled_images: List[Tuple[str, bytes]],
) -> Tuple[bytes, str]:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}"

    parts: List[Dict[str, Any]] = [{"text": prompt}]
    for label, jpeg in labeled_images:
        parts.append({"text": f"{label}:"})
        parts.append(_img_part(jpeg))

    payload = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "responseModalities": ["IMAGE"]
        },
    }

    timeout = httpx.Timeout(connect=30.0, read=300.0, write=60.0, pool=60.0)
    async with _make_client(timeout) as client:
        r = await client.post(url, json=payload, headers={"Content-Type": "application/json"})
        if r.status_code < 200 or r.status_code >= 300:
            raise HTTPException(502, f"gemini {r.status_code}: {(r.text or '')[:1500]}")
        data = r.json()

    # достаем картинку из ответа
    for cand in (data.get("candidates") or []):
        content = (cand or {}).get("content") or {}
        for part in (content.get("parts") or []):
            inline = (part or {}).get("inlineData")
            if inline and isinstance(inline, dict) and inline.get("data"):
                mime = inline.get("mimeType") or "image/png"
                return base64.b64decode(inline["data"]), mime

    raise HTTPException(502, f"gemini returned no image: {str(data)[:2000]}")


@app.post("/tryon/outfit")
async def tryon_outfit(
    top: UploadFile = File(...),
    bottom: UploadFile = File(...),
    shoes: Optional[UploadFile] = File(None),
    outer: Optional[UploadFile] = File(None),
    person_front: UploadFile = File(...),
    prompt_extra: Optional[str] = None,
):
    t_bytes = await top.read()
    b_bytes = await bottom.read()
    s_bytes = await shoes.read() if shoes else None
    o_bytes = await outer.read() if outer else None
    p_bytes = await person_front.read()

    if not t_bytes:
        raise HTTPException(400, "top is empty")
    if not b_bytes:
        raise HTTPException(400, "bottom is empty")
    if not p_bytes:
        raise HTTPException(400, "person_front is empty")

    top_img = _open_image(t_bytes)
    bottom_img = _open_image(b_bytes)
    person_img = _open_image(p_bytes)
    shoes_img = _open_image(s_bytes) if (s_bytes and len(s_bytes) > 0) else None
    outer_img = _open_image(o_bytes) if (o_bytes and len(o_bytes) > 0) else None

    # JPEG подготовка (как у тебя)
    print_crop = _make_center_crop(top_img, crop_w=0.55, crop_h=0.55)
    print_crop = _resize_max(print_crop, 1200)
    print_crop_jpeg = _encode_jpeg(print_crop, quality=92)

    top_full = _resize_max(top_img, 1600)
    top_full_jpeg = _encode_jpeg(top_full, quality=90)

    bottom_jpeg = _encode_jpeg(_resize_max(bottom_img, 1100), quality=82)
    person_jpeg = _encode_jpeg(_resize_max(person_img, 1300), quality=84)

    labeled: List[Tuple[str, bytes]] = [
        ("PRINT_CROP", print_crop_jpeg),
        ("TOP_FULL", top_full_jpeg),
        ("BOTTOM", bottom_jpeg),
    ]
    if shoes_img is not None:
        shoes_jpeg = _encode_jpeg(_resize_max(shoes_img, 1000), quality=80)
        labeled.append(("SHOES", shoes_jpeg))
    if outer_img is not None:
        outer_jpeg = _encode_jpeg(_resize_max(outer_img, 1200), quality=82)
        labeled.append(("OUTER", outer_jpeg))

    labeled.append(("PERSON_BASE", person_jpeg))  # человек всегда последним

    prompt = TRYON_OUTFIT_PROMPT
    if prompt_extra and prompt_extra.strip():
        prompt += "\n\nДоп. требования:\n" + prompt_extra.strip()

    out_bytes, mime = await _gemini_generate_image(
        model=GEMINI_MODEL,
        prompt=prompt,
        labeled_images=labeled,
    )

    return Response(content=out_bytes, media_type=mime)