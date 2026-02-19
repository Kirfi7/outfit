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

# main.py
import os
import time
import uuid
import base64
import asyncio
from io import BytesIO
from typing import Optional, List, Dict, Any, Tuple

import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response, FileResponse
from PIL import Image

# HEIC support (optional)
try:
    import pillow_heif  # type: ignore
    pillow_heif.register_heif_opener()
except Exception:
    pillow_heif = None


# ===================== CONFIG =====================
app = FastAPI()

GPTUNNEL_KEY = os.getenv("GPTUNNEL_KEY", "shds-StExbfcU2hbeyQoB1v3CGa229aV")  # не хардкодь
if not GPTUNNEL_KEY:
    raise RuntimeError("GPTUNNEL_KEY env var is not set")

GPTUNNEL_BASE = os.getenv("GPTUNNEL_BASE_URL", "https://gptunnel.ru").rstrip("/")
GPTUNNEL_CREATE_URL = f"{GPTUNNEL_BASE}/v1/media/create"
GPTUNNEL_RESULT_URL = f"{GPTUNNEL_BASE}/v1/media/result"

TRYON_MODEL_DEFAULT = os.getenv("TRYON_MODEL", "nano-banana")

TRYON_POLL_TIMEOUT_SEC = int(os.getenv("TRYON_POLL_TIMEOUT_SEC", "600"))
TRYON_POLL_INTERVAL_SEC = float(os.getenv("TRYON_POLL_INTERVAL_SEC", "1.2"))

# Публичный адрес твоего сервиса, доступный ИМ (важно!)
# пример: https://api.my-domain.com
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")
if not PUBLIC_BASE_URL:
    raise RuntimeError("PUBLIC_BASE_URL env var is not set (must be publicly reachable by GPTunnel)")

# где хранить временные файлы
TMP_DIR = os.getenv("TRYON_TMP_DIR", "/tmp/tryon_media")
os.makedirs(TMP_DIR, exist_ok=True)

# время жизни временных файлов (сек)
TMP_TTL_SEC = int(os.getenv("TRYON_TMP_TTL_SEC", "900"))

# ===================== PROMPT =====================
TRYON_OUTFIT_PROMPT = """
ЗАДАЧА: ВИРТУАЛЬНАЯ ПРИМЕРКА (ФОТО-РЕДАКТИРОВАНИЕ, НЕ генерация новой сцены).
Используй фото человека как БАЗУ и измени ТОЛЬКО одежду на нём.

ПОРЯДОК ИЗОБРАЖЕНИЙ (важно):
Image 1: TOP (футболка/верх) — каталог (целиком).
Image 2: TOP_PRINT_ZOOM — крупный план принта с Image 1 (тот же принт).
Image 3: BOTTOM (брюки/низ) — каталог.
Image 4: SHOES (обувь) — каталог, если есть.
Image 5: OUTER (куртка/верхняя одежда) — каталог, если есть.
Image 6: PERSON (человек, фронт) — БАЗОВОЕ фото. ДОЛЖНО ОСТАТЬСЯ ТЕМ ЖЕ КАДРОМ.

РАЗРЕШЕНО ИЗМЕНИТЬ ТОЛЬКО:
- одежду на человеке (верх/низ/обувь/верхняя одежда), чтобы выглядело НАДЕТО реалистично.

КРИТИЧНО ЗАПРЕЩЕНО:
- менять лицо/личность/пол/возраст/телосложение/позу/руки/ноги
- менять фон/помещение/освещение/перспективу/ракурс/кадрирование/масштаб
- добавлять других людей, манекены, новые предметы

ПРО ПРИНТ (ОЧЕНЬ ВАЖНО):
- Принт на футболке ДОЛЖЕН быть ТОЧНО таким же, как на Image 1 и Image 2:
  те же цвета, форма, расположение, пропорции, читаемость.
- НЕ перерисовывать “похожий” принт, НЕ заменять на другой, НЕ придумывать новый.
- Если принт частично закрыт курткой — оставь видимую часть принта максимально идентичной референсу.

КАК СДЕЛАТЬ:
- перенеси крой/материал/цвет TOP из Image 1, а принт — строго по Image 2 (zoom).
- реалистичная посадка, складки, тени и освещение как на базовом фото.

ВЫХОД:
одно фотореалистичное итоговое изображение.
""".strip()


# ===================== UTILS =====================
def _auth_headers() -> Dict[str, str]:
    # В доках: Authorization: YOUR_API_KEY (без Bearer)  [oai_citation:1‡GPTunnel Docs](https://docs.gptunnel.ru/media-api/image-1)
    return {"Authorization": GPTUNNEL_KEY, "Content-Type": "application/json"}


def _normalize_to_jpeg(image_bytes: bytes, max_side: int = 1100, quality: int = 82) -> bytes:
    """
    Нормализация под "ссылки" (файл будет скачивать GPTunnel).
    """
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

    w, h = img.size
    m = max(w, h)
    if m > max_side:
        scale = max_side / float(m)
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)

    out = BytesIO()
    img.save(out, format="JPEG", quality=quality, optimize=True)
    return out.getvalue()


def _center_crop(img: Image.Image, crop_factor: float = 0.55) -> Image.Image:
    """
    Кроп по центру (для TOP_PRINT_ZOOM).
    crop_factor=0.55 оставляет ~55% по ширине/высоте.
    """
    w, h = img.size
    cw, ch = int(w * crop_factor), int(h * crop_factor)
    left = max(0, (w - cw) // 2)
    top = max(0, (h - ch) // 2)
    return img.crop((left, top, left + cw, top + ch))


def _make_top_print_zoom_jpeg(top_bytes: bytes, max_side: int = 900, quality: int = 86) -> bytes:
    """
    Делаем "крупный план принта": центр + чуть выше центра (обычно принт на груди).
    """
    img = Image.open(BytesIO(top_bytes))
    try:
        from PIL import ImageOps
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass

    if img.mode != "RGB":
        img = img.convert("RGB")

    w, h = img.size
    # сместим crop чуть вверх (грудь)
    # сначала делаем crop box руками
    crop_factor = 0.55
    cw, ch = int(w * crop_factor), int(h * crop_factor)
    left = max(0, (w - cw) // 2)
    top = max(0, int((h - ch) * 0.35))  # чуть выше центра
    crop = img.crop((left, top, left + cw, top + ch))

    # resize
    ww, hh = crop.size
    m = max(ww, hh)
    if m > max_side:
        scale = max_side / float(m)
        crop = crop.resize((max(1, int(ww * scale)), max(1, int(hh * scale))), Image.Resampling.LANCZOS)

    out = BytesIO()
    crop.save(out, format="JPEG", quality=quality, optimize=True)
    return out.getvalue()


def _save_tmp(jpeg_bytes: bytes, suffix: str = ".jpg") -> Tuple[str, str]:
    """
    Возвращает (token, filepath)
    """
    token = uuid.uuid4().hex
    path = os.path.join(TMP_DIR, f"{token}{suffix}")
    with open(path, "wb") as f:
        f.write(jpeg_bytes)
    return token, path


def _pick_ar_from_person(person_bytes: bytes) -> str:
    """
    Подбираем ближайшее соотношение сторон из допустимых:
    '1:1', '16:9', '9:16', '4:3', '3:4'  [oai_citation:2‡GPTunnel Docs](https://docs.gptunnel.ru/media-api/image-1)
    """
    img = Image.open(BytesIO(person_bytes))
    w, h = img.size
    if w <= 0 or h <= 0:
        return "3:4"
    r = w / h

    candidates = {
        "1:1": 1.0,
        "4:3": 4 / 3,
        "3:4": 3 / 4,
        "16:9": 16 / 9,
        "9:16": 9 / 16,
    }
    best = min(candidates.items(), key=lambda kv: abs(kv[1] - r))
    return best[0]


async def _media_create(client: httpx.AsyncClient, prompt: str, images: List[str], model: str, ar: Optional[str]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"model": model, "prompt": prompt, "images": images}
    if ar:
        payload["ar"] = ar

    r = await client.post(GPTUNNEL_CREATE_URL, headers=_auth_headers(), json=payload)
    if r.status_code < 200 or r.status_code >= 300:
        raise HTTPException(502, f"gptunnel create {r.status_code}: {(r.text or '')[:1200]}")
    data = r.json()
    if not isinstance(data, dict) or not data.get("id"):
        raise HTTPException(502, f"unexpected create response: {str(data)[:1200]}")
    return data


async def _media_result(client: httpx.AsyncClient, task_id: str) -> Dict[str, Any]:
    r = await client.post(GPTUNNEL_RESULT_URL, headers=_auth_headers(), json={"task_id": task_id})
    if r.status_code < 200 or r.status_code >= 300:
        raise HTTPException(502, f"gptunnel result {r.status_code}: {(r.text or '')[:1200]}")
    data = r.json()
    if not isinstance(data, dict) or not data.get("id"):
        raise HTTPException(502, f"unexpected result response: {str(data)[:1200]}")
    return data


async def _wait_for_done(client: httpx.AsyncClient, task_id: str, timeout_sec: int, interval_sec: float) -> Dict[str, Any]:
    t0 = time.time()
    last: Optional[Dict[str, Any]] = None

    while True:
        last = await _media_result(client, task_id)
        status = str(last.get("status") or "").lower()
        print("tryon poll:", {"id": task_id, "status": status})

        if status == "done":
            return last
        if status in ("error", "failed"):
            raise HTTPException(502, f"tryon failed status={status}: {str(last)[:1200]}")
        if time.time() - t0 > timeout_sec:
            raise HTTPException(504, f"tryon timeout after {timeout_sec}s: {last}")

        await asyncio.sleep(interval_sec)


async def _download(url: str) -> bytes:
    async with httpx.AsyncClient(timeout=180) as c:
        r = await c.get(url)
        if r.status_code < 200 or r.status_code >= 300:
            raise HTTPException(502, f"cannot fetch result image {r.status_code}: {(r.text or '')[:300]}")
        return r.content


# ===================== TEMP FILE SERVE =====================
# GPTunnel будет забирать файлы по этим ссылкам.
# Убедись, что PUBLIC_BASE_URL указывает на домен, где доступен этот путь.
@app.get("/_tryon_media/{token}.jpg")
async def get_tryon_media(token: str):
    path = os.path.join(TMP_DIR, f"{token}.jpg")
    if not os.path.exists(path):
        raise HTTPException(404, "not found")
    return FileResponse(path, media_type="image/jpeg")


def _cleanup_old_files():
    """
    Простой best-effort cleanup по mtime.
    """
    now = time.time()
    for name in os.listdir(TMP_DIR):
        if not name.endswith(".jpg"):
            continue
        p = os.path.join(TMP_DIR, name)
        try:
            st = os.stat(p)
            if now - st.st_mtime > TMP_TTL_SEC:
                os.remove(p)
        except Exception:
            pass


# ===================== MAIN ENDPOINT =====================
@app.post("/tryon/outfit")
async def tryon_outfit(
    # вещи
    top: UploadFile = File(...),
    bottom: UploadFile = File(...),
    shoes: Optional[UploadFile] = File(None),
    outer: Optional[UploadFile] = File(None),
    # человек (база)
    person_front: UploadFile = File(...),

    prompt_extra: Optional[str] = None,
    model: Optional[str] = None,
    ar: Optional[str] = None,  # можно форсить, но по умолчанию авто
):
    _cleanup_old_files()

    # read inputs
    top_bytes = await top.read()
    bottom_bytes = await bottom.read()
    shoes_bytes = await shoes.read() if shoes else None
    outer_bytes = await outer.read() if outer else None
    person_bytes = await person_front.read()

    if not top_bytes:
        raise HTTPException(400, "top is empty")
    if not bottom_bytes:
        raise HTTPException(400, "bottom is empty")
    if not person_bytes:
        raise HTTPException(400, "person_front is empty")

    # normalize
    top_jpg = _normalize_to_jpeg(top_bytes, max_side=1100, quality=84)
    top_zoom_jpg = _make_top_print_zoom_jpeg(top_bytes)  # ключ к принту
    bottom_jpg = _normalize_to_jpeg(bottom_bytes, max_side=1100, quality=84)
    person_jpg = _normalize_to_jpeg(person_bytes, max_side=1400, quality=86)

    shoes_jpg = _normalize_to_jpeg(shoes_bytes, max_side=1100, quality=84) if shoes_bytes else None
    outer_jpg = _normalize_to_jpeg(outer_bytes, max_side=1100, quality=84) if outer_bytes else None

    # save temp + build public urls (докам нужны ссылки)  [oai_citation:3‡GPTunnel Docs](https://docs.gptunnel.ru/media-api/about)
    images_urls: List[str] = []

    def to_url(jpg_bytes: bytes) -> str:
        token, _path = _save_tmp(jpg_bytes, ".jpg")
        return f"{PUBLIC_BASE_URL}/_tryon_media/{token}.jpg"

    # ПОРЯДОК: top, top_zoom, bottom, shoes?, outer?, person(last)
    images_urls.append(to_url(top_jpg))
    images_urls.append(to_url(top_zoom_jpg))
    images_urls.append(to_url(bottom_jpg))
    if shoes_jpg:
        images_urls.append(to_url(shoes_jpg))
    if outer_jpg:
        images_urls.append(to_url(outer_jpg))
    images_urls.append(to_url(person_jpg))  # человек последним

    prompt = TRYON_OUTFIT_PROMPT
    if prompt_extra and prompt_extra.strip():
        prompt += "\n\nДоп. требования:\n" + prompt_extra.strip()

    use_model = (model or TRYON_MODEL_DEFAULT).strip()

    # ar: можно не задавать, но лучше под базовое фото человека, чтобы не «квадратило»
    use_ar = (ar or _pick_ar_from_person(person_bytes)).strip()

    async with httpx.AsyncClient(timeout=120) as client:
        created = await _media_create(
            client=client,
            prompt=prompt,
            images=images_urls,
            model=use_model,
            ar=use_ar,
        )
        task_id = created["id"]

        done = await _wait_for_done(client, task_id, TRYON_POLL_TIMEOUT_SEC, TRYON_POLL_INTERVAL_SEC)

    url = done.get("url")
    if not url:
        raise HTTPException(502, f"tryon done but url is null: {done}")

    out_bytes = await _download(str(url))
    return Response(content=out_bytes, media_type="image/png")