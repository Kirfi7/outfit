from typing import Optional, List, Tuple

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import Response

try:
    from app.core.prompts import TRYON_BASE_PASS_PROMPT, TRYON_OUTER_PASS_PROMPT
    from app.services.gemini import gemini_generate_image
    from app.services.image_utils import encode_jpeg, make_center_crop, open_image, resize_max
except ModuleNotFoundError:
    from core.prompts import TRYON_BASE_PASS_PROMPT, TRYON_OUTER_PASS_PROMPT
    from services.gemini import gemini_generate_image
    from services.image_utils import encode_jpeg, make_center_crop, open_image, resize_max

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

    print(
        "TRYON FILES bytes:",
        "top", len(t_bytes),
        "bottom", len(b_bytes),
        "shoes", (len(s_bytes) if s_bytes else None),
        "outer", (len(o_bytes) if o_bytes else None),
        "person_front", len(p_bytes),
    )

    if not t_bytes:
        raise HTTPException(400, "top is empty")
    if not b_bytes:
        raise HTTPException(400, "bottom is empty")
    if not p_bytes:
        raise HTTPException(400, "person_front is empty")

    if shoes is not None and (s_bytes is None or len(s_bytes) == 0):
        raise HTTPException(400, "shoes was provided but empty")
    if outer is not None and (o_bytes is None or len(o_bytes) == 0):
        raise HTTPException(400, "outer was provided but empty")

    top_img = open_image(t_bytes)
    bottom_img = open_image(b_bytes)
    person_img = open_image(p_bytes)

    shoes_img = open_image(s_bytes) if (s_bytes and len(s_bytes) > 0) else None
    outer_img = open_image(o_bytes) if (o_bytes and len(o_bytes) > 0) else None

    # ----- PREP -----
    # принт: делаем crop чуть крупнее, чтобы чаще попадать в принт
    print_crop = resize_max(make_center_crop(top_img, crop_w=0.70, crop_h=0.70), 1500)
    print_crop_jpeg = encode_jpeg(print_crop, quality=95)

    top_full_jpeg = encode_jpeg(resize_max(top_img, 1700), quality=93)
    bottom_jpeg = encode_jpeg(resize_max(bottom_img, 1300), quality=87)
    person_jpeg = encode_jpeg(resize_max(person_img, 1500), quality=89)

    # ---------- PASS 1 (база): TOP+PRINT+BOTTOM(+SHOES) ----------
    labeled_base: List[Tuple[str, bytes]] = [
        ("PRINT_CROP", print_crop_jpeg),
        ("TOP_FULL", top_full_jpeg),
        ("BOTTOM", bottom_jpeg),
    ]

    if shoes_img is not None:
        shoes_jpeg = encode_jpeg(resize_max(shoes_img, 1200), quality=86)
        labeled_base.append(("SHOES", shoes_jpeg))

    labeled_base.append(("PERSON_BASE", person_jpeg))

    prompt1 = TRYON_BASE_PASS_PROMPT
    if prompt_extra and prompt_extra.strip():
        prompt1 += "\n\nДоп. требования:\n" + prompt_extra.strip()

    out1_bytes, out1_mime = await gemini_generate_image(prompt1, labeled_base)

    # ---------- PASS 2 (outer): ONLY OUTER over result ----------
    if outer_img is not None:
        # out1 -> PERSON_BASE для pass2
        person2_img = open_image(out1_bytes)
        person2_jpeg = encode_jpeg(resize_max(person2_img, 1600), quality=90)

        outer_jpeg = encode_jpeg(resize_max(outer_img, 1500), quality=90)

        labeled_outer: List[Tuple[str, bytes]] = [
            ("OUTER", outer_jpeg),
            ("PERSON_BASE", person2_jpeg),
        ]

        prompt2 = TRYON_OUTER_PASS_PROMPT
        if prompt_extra and prompt_extra.strip():
            prompt2 += "\n\nДоп. требования:\n" + prompt_extra.strip()

        out2_bytes, out2_mime = await gemini_generate_image(prompt2, labeled_outer)
        return Response(content=out2_bytes, media_type=out2_mime)

    # если outer не было — возвращаем pass1
    return Response(content=out1_bytes, media_type=out1_mime)