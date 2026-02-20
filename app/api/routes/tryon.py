from typing import Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import Response

from app.core.prompts import TRYON_OUTFIT_PROMPT
from app.services.gemini import gemini_generate_image
from app.services.image_utils import encode_jpeg, make_center_crop, open_image, resize_max

router = APIRouter()


@router.post("/tryon/outfit")
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

    top_img = open_image(t_bytes)
    bottom_img = open_image(b_bytes)
    person_img = open_image(p_bytes)
    shoes_img = open_image(s_bytes) if (s_bytes and len(s_bytes) > 0) else None
    outer_img = open_image(o_bytes) if (o_bytes and len(o_bytes) > 0) else None

    print_crop = resize_max(make_center_crop(top_img, crop_w=0.62, crop_h=0.62), 1400)
    print_crop_jpeg = encode_jpeg(print_crop, quality=94)
    top_full_jpeg = encode_jpeg(resize_max(top_img, 1600), quality=92)
    bottom_jpeg = encode_jpeg(resize_max(bottom_img, 1200), quality=86)
    person_jpeg = encode_jpeg(resize_max(person_img, 1400), quality=88)

    labeled = [
        ("PRINT_CROP", print_crop_jpeg),
        ("TOP_FULL", top_full_jpeg),
        ("BOTTOM", bottom_jpeg),
    ]

    if shoes_img is not None:
        labeled.append(("SHOES", encode_jpeg(resize_max(shoes_img, 1100), quality=84)))
    if outer_img is not None:
        labeled.append(("OUTER", encode_jpeg(resize_max(outer_img, 1300), quality=86)))

    labeled.append(("PERSON_BASE", person_jpeg))

    prompt = TRYON_OUTFIT_PROMPT
    if prompt_extra and prompt_extra.strip():
        prompt += "\n\nДоп. требования:\n" + prompt_extra.strip()

    out_bytes, mime = await gemini_generate_image(prompt, labeled)
    return Response(content=out_bytes, media_type=mime)
