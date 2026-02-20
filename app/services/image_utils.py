import base64
from io import BytesIO

from fastapi import HTTPException
from PIL import Image

try:
    import pillow_heif  # type: ignore

    pillow_heif.register_heif_opener()
except Exception:
    pillow_heif = None


def open_image(image_bytes: bytes) -> Image.Image:
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


def resize_max(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    m = max(w, h)
    if m <= max_side:
        return img
    scale = max_side / float(m)
    return img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)


def encode_jpeg(img: Image.Image, quality: int = 82) -> bytes:
    out = BytesIO()
    img.save(out, format="JPEG", quality=quality, optimize=True)
    return out.getvalue()


def to_data_url_jpeg(jpeg_bytes: bytes) -> str:
    b64 = base64.b64encode(jpeg_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def make_center_crop(img: Image.Image, crop_w: float = 0.62, crop_h: float = 0.62) -> Image.Image:
    w, h = img.size
    cw = int(w * crop_w)
    ch = int(h * crop_h)
    left = max(0, (w - cw) // 2)
    top = max(0, (h - ch) // 2)
    return img.crop((left, top, left + cw, top + ch))
