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
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import Response
from PIL import Image

# HEIC support (optional)
try:
    import pillow_heif  # type: ignore
    pillow_heif.register_heif_opener()
except Exception:
    pillow_heif = None


app = FastAPI()

# ============ GPTUNNEL CreativeLab config ============
GPTUNNEL_KEY = os.getenv("GPTUNNEL_KEY", "shds-StExbfcU2hbeyQoB1v3CGa229aV")
if not GPTUNNEL_KEY:
    raise RuntimeError("GPTUNNEL_KEY env var is not set")

GPTUNNEL_BASE = os.getenv("GPTUNNEL_BASE_URL", "https://gptunnel.ru").rstrip("/")
GPTUNNEL_CREATE_URL = f"{GPTUNNEL_BASE}/v1/media/create"
GPTUNNEL_RESULT_URL = f"{GPTUNNEL_BASE}/v1/media/result"

TRYON_MODEL_DEFAULT = os.getenv("TRYON_MODEL", "nano-banana-2")

TRYON_POLL_TIMEOUT_SEC = int(os.getenv("TRYON_POLL_TIMEOUT_SEC", "600"))
TRYON_POLL_INTERVAL_SEC = float(os.getenv("TRYON_POLL_INTERVAL_SEC", "1.2"))

ALLOWED_AR = ["1:1", "16:9", "9:16", "4:3", "3:4"]

# ---------------- PROMPT ----------------
TRYON_OUTFIT_PROMPT = """
ЗАДАЧА: ФОТО-РЕДАКТИРОВАНИЕ (НЕ генерация новой сцены).
Нужно отредактировать базовое фото человека (ПОСЛЕДНИЙ Image), изменив ТОЛЬКО его одежду.

ПОРЯДОК ИЗОБРАЖЕНИЙ (КРИТИЧНО):
- Image 1: PRINT_CROP (крупный план принта на футболке/верху) — ЭТАЛОН ПРИНТА. Должен совпасть 1:1.
- Image 2: TOP_FULL (полное фото верха) — цвет/крой/посадка верха.
- Image 3: BOTTOM (низ)
- Image 4: SHOES (если есть)
- Image 5: OUTER (если есть)
- Последний Image: PERSON_BASE (человек) — БАЗОВОЕ ФОТО, его нельзя менять кроме одежды.

РАЗРЕШЕНО МЕНЯТЬ ТОЛЬКО:
- верх/низ/обувь/верхнюю одежду на человеке.

ЗАПРЕЩЕНО (СТРОГО):
- менять лицо/личность/пол/возраст/бороду/прическу
- менять телосложение/рост/пропорции/позу/положение рук и ног
- менять фон/помещение/мебель/свет/перспективу/ракурс
- менять кадрирование/масштаб (не приближать, не переносить в другую комнату)
- добавлять других людей/предметы

КРИТИЧНО ПРО ПРИНТ:
- Принт на груди должен точно соответствовать Image 1 (PRINT_CROP): та же форма, контуры, цвета, расположение.
- Не перерисовывай принт "по мотивам" и не подменяй текст/картинку.

ЕСЛИ чего-то нет:
- Если SHOES или OUTER не переданы — НЕ добавляй их.

ВЫХОД:
Одно фотореалистичное изображение: тот же человек/фон, но в предоставленных вещах.
""".strip()


def _auth_headers() -> Dict[str, str]:
    # По докам GPTunnel: Authorization: KEY (без Bearer)
    return {"Authorization": GPTUNNEL_KEY, "Content-Type": "application/json"}


def _open_image(image_bytes: bytes) -> Image.Image:
    try:
        img = Image.open(BytesIO(image_bytes))
    except Exception as e:
        hint = " (HEIC? install pillow-heif)" if pillow_heif is None else ""
        raise HTTPException(400, f"Cannot decode image{hint}: {e}")

    # EXIF orientation fix
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


def _to_data_url_jpeg(jpeg_bytes: bytes) -> str:
    b64 = base64.b64encode(jpeg_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def _make_center_crop(img: Image.Image, crop_w: float = 0.55, crop_h: float = 0.55) -> Image.Image:
    """
    Простой кроп по центру (для каталожных футболок на белом фоне это обычно попадает в принт).
    """
    w, h = img.size
    cw = int(w * crop_w)
    ch = int(h * crop_h)
    left = max(0, (w - cw) // 2)
    top = max(0, (h - ch) // 2)
    return img.crop((left, top, left + cw, top + ch))


def _pick_ar_from_size(w: int, h: int) -> str:
    r = w / float(h)
    # близко к квадрату
    if 0.9 <= r <= 1.1:
        return "1:1"
    # портрет
    if r < 1.0:
        # 3:4 = 0.75, 9:16 = 0.5625
        return "9:16" if r < 0.67 else "3:4"
    # ландшафт
    # 4:3 = 1.333, 16:9 = 1.777
    return "16:9" if r > 1.55 else "4:3"


def _decode_data_url_image(url: str) -> Optional[bytes]:
    url = (url or "").strip()
    if not url.startswith("data:"):
        return None
    try:
        b64_part = url.split(",", 1)[1]
        return base64.b64decode(b64_part)
    except Exception:
        return None


async def _fetch_image_bytes(url: str, client: httpx.AsyncClient) -> bytes:
    maybe = _decode_data_url_image(url)
    if maybe:
        return maybe
    r = await client.get(url)
    if r.status_code < 200 or r.status_code >= 300:
        raise HTTPException(502, f"cannot fetch result image {r.status_code}: {(r.text or '')[:300]}")
    return r.content


async def _media_create(payload: Dict[str, Any], client: httpx.AsyncClient) -> Dict[str, Any]:
    r = await client.post(GPTUNNEL_CREATE_URL, headers=_auth_headers(), json=payload)
    if r.status_code < 200 or r.status_code >= 300:
        raise HTTPException(502, f"gptunnel create {r.status_code}: {(r.text or '')[:1200]}")
    data = r.json()
    if not isinstance(data, dict) or not data.get("id"):
        raise HTTPException(502, f"unexpected create response: {str(data)[:1200]}")
    return data


async def _media_result(task_id: str, client: httpx.AsyncClient) -> Dict[str, Any]:
    r = await client.post(GPTUNNEL_RESULT_URL, headers=_auth_headers(), json={"task_id": task_id})
    if r.status_code < 200 or r.status_code >= 300:
        raise HTTPException(502, f"gptunnel result {r.status_code}: {(r.text or '')[:1200]}")
    data = r.json()
    if not isinstance(data, dict) or not data.get("id"):
        raise HTTPException(502, f"unexpected result response: {str(data)[:1200]}")
    return data


async def _wait_for_done(task_id: str, timeout_sec: int, interval_sec: float, client: httpx.AsyncClient) -> Dict[str, Any]:
    t0 = time.time()
    last: Optional[Dict[str, Any]] = None

    while True:
        last = await _media_result(task_id, client)
        status = str(last.get("status") or "").lower()

        # Логи на сервер (полезно)
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
    # вещи
    top: UploadFile = File(...),
    bottom: UploadFile = File(...),
    shoes: Optional[UploadFile] = File(None),
    outer: Optional[UploadFile] = File(None),
    # человек (БАЗА)
    person_front: UploadFile = File(...),

    prompt_extra: Optional[str] = None,
    model: Optional[str] = None,
    ar: Optional[str] = Query(default=None, description="1:1|16:9|9:16|4:3|3:4"),
):
    # -------- read inputs
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

    # -------- decode images
    top_img = _open_image(t_bytes)
    bottom_img = _open_image(b_bytes)
    person_img = _open_image(p_bytes)
    shoes_img = _open_image(s_bytes) if (s_bytes and len(s_bytes) > 0) else None
    outer_img = _open_image(o_bytes) if (o_bytes and len(o_bytes) > 0) else None

    # -------- ar
    auto_ar = _pick_ar_from_size(*person_img.size)
    use_ar = (ar or auto_ar).strip()
    if use_ar not in ALLOWED_AR:
        raise HTTPException(400, f"ar must be one of {ALLOWED_AR}, got {use_ar}")

    # -------- encode with DIFFERENT quality
    # PRINT_CROP: максимум качества
    print_crop = _make_center_crop(top_img, crop_w=0.55, crop_h=0.55)
    print_crop = _resize_max(print_crop, 1200)
    print_crop_jpeg = _encode_jpeg(print_crop, quality=92)

    # TOP_FULL: высокое качество
    top_full = _resize_max(top_img, 1600)
    top_full_jpeg = _encode_jpeg(top_full, quality=90)

    # остальное можно пожать сильнее
    bottom_jpeg = _encode_jpeg(_resize_max(bottom_img, 1100), quality=82)
    person_jpeg = _encode_jpeg(_resize_max(person_img, 1300), quality=84)

    images: List[str] = []
    images.append(_to_data_url_jpeg(print_crop_jpeg))   # Image 1 PRINT_CROP (эталон принта)
    images.append(_to_data_url_jpeg(top_full_jpeg))     # Image 2 TOP_FULL
    images.append(_to_data_url_jpeg(bottom_jpeg))       # Image 3 BOTTOM

    if shoes_img is not None:
        shoes_jpeg = _encode_jpeg(_resize_max(shoes_img, 1000), quality=80)
        images.append(_to_data_url_jpeg(shoes_jpeg))    # Image 4 SHOES
    if outer_img is not None:
        outer_jpeg = _encode_jpeg(_resize_max(outer_img, 1200), quality=82)
        images.append(_to_data_url_jpeg(outer_jpeg))    # Image 5 OUTER

    # КЛЮЧЕВО: человек всегда последним
    images.append(_to_data_url_jpeg(person_jpeg))       # LAST = PERSON_BASE

    # -------- prompt
    prompt = TRYON_OUTFIT_PROMPT
    if prompt_extra and prompt_extra.strip():
        prompt += "\n\nДоп. требования:\n" + prompt_extra.strip()

    use_model = (model or TRYON_MODEL_DEFAULT).strip()

    payload: Dict[str, Any] = {
        "model": use_model,
        "prompt": prompt,
        "images": images,
        "ar": use_ar,
    }

    # -------- call gptunnel
    timeout = httpx.Timeout(connect=30.0, read=180.0, write=60.0, pool=60.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        created = await _media_create(payload, client)
        task_id = created["id"]

        done = await _wait_for_done(task_id, TRYON_POLL_TIMEOUT_SEC, TRYON_POLL_INTERVAL_SEC, client)

        url = done.get("url")
        if not url:
            raise HTTPException(502, f"tryon done but url is null: {done}")

        out_bytes = await _fetch_image_bytes(str(url), client)

    # обычно webp/png — но клиенту ок отдавать как image/png, браузер откроет
    return Response(content=out_bytes, media_type="image/png")