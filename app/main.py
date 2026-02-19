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

import os
import time
import base64
import asyncio
from io import BytesIO
from typing import Optional, List, Dict, Any, Tuple

import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
from PIL import Image

try:
    import pillow_heif  # type: ignore
    pillow_heif.register_heif_opener()
except Exception:
    pillow_heif = None

app = FastAPI()

GPTUNNEL_KEY = os.getenv("GPTUNNEL_KEY", "")
if not GPTUNNEL_KEY:
    raise RuntimeError("GPTUNNEL_KEY env var is not set")

GPTUNNEL_BASE = os.getenv("GPTUNNEL_BASE_URL", "https://gptunnel.ru")
GPTUNNEL_CREATE_URL = f"{GPTUNNEL_BASE}/v1/media/create"
GPTUNNEL_RESULT_URL = f"{GPTUNNEL_BASE}/v1/media/result"

TRYON_MODEL_DEFAULT = os.getenv("TRYON_MODEL", "nano-banana")
TRYON_POLL_TIMEOUT_SEC = int(os.getenv("TRYON_POLL_TIMEOUT_SEC", "600"))
TRYON_POLL_INTERVAL_SEC = float(os.getenv("TRYON_POLL_INTERVAL_SEC", "1.2"))

TRYON_OUTFIT_PROMPT = """
ЗАДАЧА: ФОТО-РЕДАКТИРОВАНИЕ (НЕ генерация новой сцены).
Используй базовое фото человека и измени ТОЛЬКО одежду.

ПОРЯДОК ВХОДА (КРИТИЧНО):
- Последнее изображение (последний Image) — БАЗОВОЕ фото ЧЕЛОВЕКА.
- Остальные изображения до него — РЕФЕРЕНСЫ ВЕЩЕЙ.

ВХОД:
Image 1: верх (top) — каталожное фото футболки/верха. ЭТО АВТОРИТЕТНЫЙ ИСТОЧНИК ПРИНТА/ЛОГО/ТЕКСТА.
Image 2: низ (bottom)
Image 3: обувь (shoes), если есть
Image 4: верхняя одежда (outer), если есть
Image 5: человек (base) — ПОСЛЕДНИЙ image

РАЗРЕШЕНО ИЗМЕНИТЬ ТОЛЬКО:
- одежду на человеке (верх/низ/обувь/outer)

ЗАПРЕЩЕНО:
- менять лицо/личность/пол/возраст/телосложение/позу
- менять фон/свет/перспективу/ракурс/кадрирование
- добавлять других людей/манекены/предметы

КРИТИЧНО ПРО ВЕРХ (футболку):
- принт/надписи/логотипы должны совпадать с Image 1 (top) максимально точно
- НЕ упрощать принт, НЕ заменять на другой, НЕ рисовать "похожий"
- если принт на Image 1 виден плохо — всё равно НЕ выдумывать новый, лучше оставить максимально близко как на Image 1

ВЫХОД:
одно фотореалистичное изображение.
""".strip()


def _auth_headers() -> Dict[str, str]:
    return {
        "Authorization": GPTUNNEL_KEY,  # без Bearer
        "Content-Type": "application/json",
    }


def _open_image(image_bytes: bytes) -> Image.Image:
    try:
        img = Image.open(BytesIO(image_bytes))
    except Exception as e:
        hint = ""
        if pillow_heif is None:
            hint = " (HEIC? install pillow-heif)"
        raise HTTPException(400, f"Cannot decode image{hint}: {e}")

    try:
        from PIL import ImageOps
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass

    return img


def _resize(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    m = max(w, h)
    if m <= max_side:
        return img
    scale = max_side / float(m)
    return img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)


def _to_jpeg_bytes(img: Image.Image, max_side: int, quality: int) -> bytes:
    img = _resize(img, max_side)
    if img.mode != "RGB":
        img = img.convert("RGB")
    out = BytesIO()
    img.save(out, format="JPEG", quality=quality, optimize=True)
    return out.getvalue()


def _to_png_bytes(img: Image.Image, max_side: int) -> bytes:
    img = _resize(img, max_side)
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGBA")
    out = BytesIO()
    img.save(out, format="PNG", optimize=True)
    return out.getvalue()


def _to_data_url(data: bytes, mime: str) -> str:
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _decode_data_url_image(url: str) -> Optional[bytes]:
    url = (url or "").strip()
    if not url.startswith("data:"):
        return None
    try:
        return base64.b64decode(url.split(",", 1)[1])
    except Exception:
        return None


async def _fetch_image_bytes(url: str) -> bytes:
    maybe = _decode_data_url_image(url)
    if maybe:
        return maybe
    async with httpx.AsyncClient(timeout=180) as client:
        r = await client.get(url)
        if r.status_code < 200 or r.status_code >= 300:
            raise HTTPException(502, f"cannot fetch result image {r.status_code}: {(r.text or '')[:300]}")
        return r.content


async def _media_create(prompt: str, images: List[str], model: str) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"model": model, "prompt": prompt, "images": images}
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(GPTUNNEL_CREATE_URL, headers=_auth_headers(), json=payload)
        if r.status_code < 200 or r.status_code >= 300:
            raise HTTPException(502, f"gptunnel create {r.status_code}: {(r.text or '')[:1200]}")
        data = r.json()
    if not isinstance(data, dict) or not data.get("id"):
        raise HTTPException(502, f"unexpected create response: {str(data)[:1200]}")
    return data


async def _media_result(task_id: str) -> Dict[str, Any]:
    payload = {"task_id": task_id}
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(GPTUNNEL_RESULT_URL, headers=_auth_headers(), json=payload)
        if r.status_code < 200 or r.status_code >= 300:
            raise HTTPException(502, f"gptunnel result {r.status_code}: {(r.text or '')[:1200]}")
        data = r.json()
    if not isinstance(data, dict) or not data.get("id"):
        raise HTTPException(502, f"unexpected result response: {str(data)[:1200]}")
    return data


async def _wait_for_done(task_id: str, timeout_sec: int, interval_sec: float) -> Dict[str, Any]:
    t0 = time.time()
    last: Optional[Dict[str, Any]] = None
    while True:
        last = await _media_result(task_id)
        status = str(last.get("status") or "").lower()
        print("tryon poll:", {"id": task_id, "status": status})

        if status == "done":
            return last
        if status in ("error", "failed"):
            raise HTTPException(502, f"tryon failed status={status}: {str(last)[:1200]}")
        if time.time() - t0 > timeout_sec:
            raise HTTPException(504, f"tryon timeout after {timeout_sec}s: {last}")

        await asyncio.sleep(interval_sec)


@app.post("/tryon/outfit")
async def tryon_outfit(
    top: UploadFile = File(...),
    bottom: UploadFile = File(...),
    shoes: Optional[UploadFile] = File(None),
    outer: Optional[UploadFile] = File(None),
    person_front: UploadFile = File(...),
    prompt_extra: Optional[str] = None,
    model: Optional[str] = None,
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

    # --- decode images ---
    top_img = _open_image(t_bytes)
    bottom_img = _open_image(b_bytes)
    person_img = _open_image(p_bytes)

    shoes_img = _open_image(s_bytes) if (s_bytes and len(s_bytes) > 0) else None
    outer_img = _open_image(o_bytes) if (o_bytes and len(o_bytes) > 0) else None

    # --- IMPORTANT: keep TOP high fidelity (PNG, bigger) ---
    top_png = _to_png_bytes(top_img, max_side=1400)   # принт не разваливается
    bottom_jpg = _to_jpeg_bytes(bottom_img, max_side=1100, quality=84)
    person_jpg = _to_jpeg_bytes(person_img, max_side=900, quality=72)

    images: List[str] = []
    images.append(_to_data_url(top_png, "image/png"))                 # top
    images.append(_to_data_url(bottom_jpg, "image/jpeg"))             # bottom

    if shoes_img is not None:
        shoes_jpg = _to_jpeg_bytes(shoes_img, max_side=1100, quality=84)
        images.append(_to_data_url(shoes_jpg, "image/jpeg"))

    if outer_img is not None:
        outer_jpg = _to_jpeg_bytes(outer_img, max_side=1100, quality=84)
        images.append(_to_data_url(outer_jpg, "image/jpeg"))

    # человек — СТРОГО последним (как у тебя “вручную” работало)
    images.append(_to_data_url(person_jpg, "image/jpeg"))

    prompt = TRYON_OUTFIT_PROMPT
    if prompt_extra and prompt_extra.strip():
        prompt += "\n\nДоп. требования:\n" + prompt_extra.strip()

    use_model = (model or TRYON_MODEL_DEFAULT).strip()

    created = await _media_create(prompt=prompt, images=images, model=use_model)
    task_id = created["id"]

    done = await _wait_for_done(task_id, TRYON_POLL_TIMEOUT_SEC, TRYON_POLL_INTERVAL_SEC)
    url = done.get("url")
    if not url:
        raise HTTPException(502, f"tryon done but url is null: {done}")

    out_bytes = await _fetch_image_bytes(str(url))
    return Response(content=out_bytes, media_type="image/png")