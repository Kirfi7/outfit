# main.py
import os
import re
import json
import base64
from io import BytesIO
from typing import Optional, Any, Dict, List, Tuple

import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from PIL import Image

# HEIC support (optional)
try:
    import pillow_heif  # type: ignore
    pillow_heif.register_heif_opener()
except Exception:
    pillow_heif = None


# =============================================================================
# 0) LOAD ONLY KEYS FROM /home/admin/api.txt
# =============================================================================
API_TXT_PATH = "/home/admin/api.txt"


def load_keys_from_txt(path: str) -> None:
    """
    Reads KEY=VALUE lines and sets os.environ[KEY]=VALUE.
    Ignores empty lines and comments (#...).
    """
    if not os.path.exists(path):
        raise RuntimeError(f"api.txt not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k and v:
                os.environ[k] = v


load_keys_from_txt(API_TXT_PATH)

# EXPECTED KEYS IN TXT:
# AITUNNEL_KEY=...
# GEMINI_API_KEY=...


# =============================================================================
# 1) HARD-CODED CONFIG (оставляем в коде как ты просил)
# =============================================================================

# ---------- AITunnel (sizes) ----------
AITUNNEL_URL = "https://api.aitunnel.ru/v1/chat/completions"
AITUNNEL_MODEL = "gemini-2.5-flash-lite"  # можешь поменять тут
AITUNNEL_KEY = os.getenv("AITUNNEL_KEY", "").strip()
if not AITUNNEL_KEY:
    raise RuntimeError("AITUNNEL_KEY is missing in /home/admin/api.txt")

# ---------- Google Gemini (try-on) ----------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing in /home/admin/api.txt")

GEMINI_MODEL = "gemini-2.5-flash-image"  # или gemini-2.5-pro-image / что у тебя работает
IMAGE_AR = "16:9"                        # "1:1" / "9:16" / etc.
IMAGE_SIZE = "1K"                        # "1K" | "2K" | "4K"
PROXY_URL = "http://sve2Zh:1W4P9D@138.219.173.226:8000"                           # если надо — впиши сюда http://login:pass@ip:port

# =============================================================================
# 2) PROMPTS
# =============================================================================
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
""".strip()

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
""".strip()

TRYON_OUTFIT_PROMPT = """
ЗАДАЧА: ФОТО-РЕДАКТИРОВАНИЕ (НЕ генерация новой сцены).
Нужно отредактировать базовое фото человека (PERSON_BASE), изменив ТОЛЬКО его одежду.

ПОРЯДОК ИЗОБРАЖЕНИЙ (КРИТИЧНО):
1) PRINT_CROP — крупный план принта (ЭТАЛОН принта, совпадение 1:1)
2) TOP_FULL — полное каталожное фото верха (цвет/крой/материал)
3) BOTTOM — низ
4) SHOES — обувь (если есть)
5) OUTER — верхняя одежда (если есть)
6) PERSON_BASE — человек (базовое фото), его нельзя менять кроме одежды

РАЗРЕШЕНО МЕНЯТЬ ТОЛЬКО:
- одежду на человеке (верх/низ/обувь/верхняя одежда).

ЗАПРЕЩЕНО (СТРОГО):
- менять лицо/личность/пол/возраст/прическу/бороду
- менять телосложение/рост/пропорции/позу
- менять фон/свет/перспективу/ракурс/кадрирование
- добавлять других людей/предметы

КРИТИЧНО ПРО ПРИНТ:
- Принт на груди должен ТОЧНО соответствовать PRINT_CROP:
  форма/контуры/цвета/расположение/текст.
- Нельзя перерисовывать "по мотивам" и нельзя подменять принт другим.

ЕСЛИ SHOES или OUTER не переданы — НЕ добавляй их.

ВЫХОД:
Одно фотореалистичное изображение: тот же человек/фон, но в предоставленных вещах.
""".strip()


# =============================================================================
# 3) FastAPI + Models
# =============================================================================
app = FastAPI()


class TshirtOut(BaseModel):
    ru: Optional[int] = None
    intl: Optional[str] = None


class PantsOut(BaseModel):
    ru: Optional[int] = None
    waist_in: Optional[int] = None


class ClothesOut(BaseModel):
    head_cm: Optional[int] = None
    tshirt: TshirtOut
    pants: PantsOut
    confidence: float
    notes: str


class ShoesOut(BaseModel):
    shoe_eu: Optional[int] = None
    confidence: float
    notes: str


# =============================================================================
# 4) Helpers
# =============================================================================
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


def _safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        s = str(x).strip()
        if s.endswith(".0"):
            s = s[:-2]
        return int(s)
    except Exception:
        return None


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


def _encode_jpeg(img: Image.Image, quality: int = 82) -> bytes:
    out = BytesIO()
    img.save(out, format="JPEG", quality=quality, optimize=True)
    return out.getvalue()


def _to_data_url_jpeg(jpeg_bytes: bytes) -> str:
    b64 = base64.b64encode(jpeg_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def _make_center_crop(img: Image.Image, crop_w: float = 0.62, crop_h: float = 0.62) -> Image.Image:
    w, h = img.size
    cw = int(w * crop_w)
    ch = int(h * crop_h)
    left = max(0, (w - cw) // 2)
    top = max(0, (h - ch) // 2)
    return img.crop((left, top, left + cw, top + ch))


# =============================================================================
# 5) AITUNNEL calls (async)
# =============================================================================
async def _aitunnel_chat(messages: Any) -> Dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {AITUNNEL_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": AITUNNEL_MODEL,
        "messages": messages,
        "temperature": 0,
        "top_p": 1,
        "max_tokens": 600,
    }

    timeout = httpx.Timeout(connect=20.0, read=120.0, write=60.0, pool=60.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(AITUNNEL_URL, headers=headers, json=payload)

    if r.status_code != 200:
        raise HTTPException(502, f"aitunnel error {r.status_code}: {(r.text or '')[:1500]}")

    data = r.json()
    try:
        text = data["choices"][0]["message"]["content"]
    except Exception:
        raise HTTPException(502, f"unexpected aitunnel response: {str(data)[:1500]}")

    try:
        return _extract_json(text)
    except Exception as e:
        raise HTTPException(502, f"bad model json: {e}; raw={(str(text)[:500])}")


# =============================================================================
# 6) GEMINI generateContent (async)
# =============================================================================
def _img_part(jpeg_bytes: bytes) -> Dict[str, Any]:
    b64 = base64.b64encode(jpeg_bytes).decode("utf-8")
    return {"inlineData": {"mimeType": "image/jpeg", "data": b64}}


def _make_httpx_client(timeout: httpx.Timeout) -> httpx.AsyncClient:
    if not PROXY_URL:
        return httpx.AsyncClient(timeout=timeout, trust_env=False)
    try:
        return httpx.AsyncClient(timeout=timeout, trust_env=False, proxy=PROXY_URL)
    except TypeError:
        return httpx.AsyncClient(timeout=timeout, trust_env=False, proxies={"http": PROXY_URL, "https": PROXY_URL})


async def _gemini_generate_image(prompt: str, labeled_images: List[Tuple[str, bytes]]) -> Tuple[bytes, str]:
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    )

    parts: List[Dict[str, Any]] = [{"text": prompt}]
    for label, jpeg in labeled_images:
        parts.append({"text": f"{label}:"})
        parts.append(_img_part(jpeg))

    payload = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "imageConfig": {
                "aspectRatio": IMAGE_AR,
                "imageSize": IMAGE_SIZE,
            }
        },
    }

    timeout = httpx.Timeout(connect=30.0, read=600.0, write=60.0, pool=60.0)
    async with _make_httpx_client(timeout) as client:
        r = await client.post(url, json=payload, headers={"Content-Type": "application/json"})

    if r.status_code < 200 or r.status_code >= 300:
        raise HTTPException(502, f"gemini {r.status_code}: {(r.text or '')[:1500]}")

    data = r.json()

    # parse image from response
    for cand in (data.get("candidates") or []):
        content = (cand or {}).get("content") or {}
        for part in (content.get("parts") or []):
            inline = (part or {}).get("inlineData")
            if inline and inline.get("data"):
                mime = inline.get("mimeType") or "image/png"
                return base64.b64decode(inline["data"]), mime

    raise HTTPException(502, f"gemini returned no image: {str(data)[:2000]}")


# =============================================================================
# 7) ENDPOINTS
# =============================================================================
@app.post("/sizes/clothes", response_model=ClothesOut)
async def sizes_clothes(
    front: UploadFile = File(...),
    back: UploadFile = File(...),
):
    front_bytes = await front.read()
    back_bytes = await back.read()
    if not front_bytes or not back_bytes:
        raise HTTPException(400, "front/back images required")

    front_img = _resize_max(_open_image(front_bytes), 1280)
    back_img = _resize_max(_open_image(back_bytes), 1280)

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": CLOTHES_PROMPT},
            {"type": "image_url", "image_url": {"url": _to_data_url_jpeg(_encode_jpeg(front_img, 82))}},
            {"type": "image_url", "image_url": {"url": _to_data_url_jpeg(_encode_jpeg(back_img, 82))}},
        ],
    }]

    j = await _aitunnel_chat(messages)

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

    side_img = _resize_max(_open_image(side_bytes), 1280)

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": SHOES_PROMPT},
            {"type": "image_url", "image_url": {"url": _to_data_url_jpeg(_encode_jpeg(side_img, 82))}},
        ],
    }]

    j = await _aitunnel_chat(messages)

    out = {
        "shoe_eu": _safe_int(j.get("shoe_eu")),
        "confidence": float(j.get("confidence") or 0.0),
        "notes": str(j.get("notes") or ""),
    }
    return ShoesOut(**out)


@app.post("/tryon/outfit")
async def tryon_outfit(
    top: UploadFile = File(...),
    bottom: UploadFile = File(...),
    shoes: Optional[UploadFile] = File(None),
    outer: Optional[UploadFile] = File(None),
    person_front: UploadFile = File(...),
    prompt_extra: Optional[str] = None,
):
    """
    Try-on через Google Gemini (generateContent):
    - PRINT_CROP делаем из top, чтобы модель видела принт максимально чётко
    - TOP_FULL даём целиком изделие
    - человек всегда последним (PERSON_BASE)
    """
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

    # Принт — максимальное качество
    print_crop = _make_center_crop(top_img, crop_w=0.62, crop_h=0.62)
    print_crop = _resize_max(print_crop, 1400)
    print_crop_jpeg = _encode_jpeg(print_crop, quality=94)

    # Верх целиком
    top_full = _resize_max(top_img, 1600)
    top_full_jpeg = _encode_jpeg(top_full, quality=92)

    bottom_jpeg = _encode_jpeg(_resize_max(bottom_img, 1200), quality=86)
    person_jpeg = _encode_jpeg(_resize_max(person_img, 1400), quality=88)

    labeled: List[Tuple[str, bytes]] = [
        ("PRINT_CROP", print_crop_jpeg),
        ("TOP_FULL", top_full_jpeg),
        ("BOTTOM", bottom_jpeg),
    ]

    if shoes_img is not None:
        labeled.append(("SHOES", _encode_jpeg(_resize_max(shoes_img, 1100), quality=84)))
    if outer_img is not None:
        labeled.append(("OUTER", _encode_jpeg(_resize_max(outer_img, 1300), quality=86)))

    labeled.append(("PERSON_BASE", person_jpeg))  # человек строго последним

    prompt = TRYON_OUTFIT_PROMPT
    if prompt_extra and prompt_extra.strip():
        prompt += "\n\nДоп. требования:\n" + prompt_extra.strip()

    out_bytes, mime = await _gemini_generate_image(prompt, labeled)
    return Response(content=out_bytes, media_type=mime)