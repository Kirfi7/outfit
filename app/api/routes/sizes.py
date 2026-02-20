from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core.prompts import CLOTHES_PROMPT, SHOES_PROMPT
from app.schemas.sizes import ClothesOut, ShoesOut
from app.services.aitunnel import aitunnel_chat
from app.services.image_utils import encode_jpeg, open_image, resize_max, to_data_url_jpeg
from app.services.parsing import safe_int

router = APIRouter()


@router.post("/sizes/clothes", response_model=ClothesOut)
async def sizes_clothes(front: UploadFile = File(...), back: UploadFile = File(...)):
    front_bytes = await front.read()
    back_bytes = await back.read()
    if not front_bytes or not back_bytes:
        raise HTTPException(400, "front/back images required")

    front_img = resize_max(open_image(front_bytes), 1280)
    back_img = resize_max(open_image(back_bytes), 1280)

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": CLOTHES_PROMPT},
            {"type": "image_url", "image_url": {"url": to_data_url_jpeg(encode_jpeg(front_img, 82))}},
            {"type": "image_url", "image_url": {"url": to_data_url_jpeg(encode_jpeg(back_img, 82))}},
        ],
    }]

    j = await aitunnel_chat(messages)
    tshirt = j.get("tshirt") or {}
    pants = j.get("pants") or {}

    out = {
        "head_cm": safe_int(j.get("head_cm")),
        "tshirt": {"ru": safe_int(tshirt.get("ru")), "intl": tshirt.get("intl") or None},
        "pants": {"ru": safe_int(pants.get("ru")), "waist_in": safe_int(pants.get("waist_in"))},
        "confidence": float(j.get("confidence") or 0.0),
        "notes": str(j.get("notes") or ""),
    }
    return ClothesOut(**out)


@router.post("/sizes/shoes", response_model=ShoesOut)
async def sizes_shoes(side: UploadFile = File(...)):
    side_bytes = await side.read()
    if not side_bytes:
        raise HTTPException(400, "side image required")

    side_img = resize_max(open_image(side_bytes), 1280)

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": SHOES_PROMPT},
            {"type": "image_url", "image_url": {"url": to_data_url_jpeg(encode_jpeg(side_img, 82))}},
        ],
    }]

    j = await aitunnel_chat(messages)

    out = {
        "shoe_eu": safe_int(j.get("shoe_eu")),
        "confidence": float(j.get("confidence") or 0.0),
        "notes": str(j.get("notes") or ""),
    }
    return ShoesOut(**out)
