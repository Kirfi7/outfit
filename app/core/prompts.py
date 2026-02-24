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

# ---- PASS 1: собираем базовый лук БЕЗ outer ----
TRYON_BASE_PASS_PROMPT = """
TASK: PHOTO EDITING (NOT generating a new scene).
Edit PERSON_BASE by changing ONLY the clothing: TOP_FULL + PRINT_CROP print + BOTTOM (+ SHOES if provided).
The result must look like the same photo: same person, same background.

STRICT RULES:
- PERSON_BASE is the only base image. Do NOT change the face, identity, age, gender, body shape, pose, background, lighting, camera angle, or framing.
- You may change ONLY the clothing on the person (top/bottom/shoes).
- IN THIS PASS you must NOT add/change outerwear (OUTER). Even if it is provided — ignore it.

INPUTS (by tags):
PRINT_CROP = the exact chest print reference (1:1).
TOP_FULL   = catalog photo of the top (t-shirt/polo): color/cut/material.
BOTTOM     = bottom (pants/jeans): color/cut/material.
SHOES      = shoes (if provided).
PERSON_BASE= the person (base).

PRINT IS CRITICAL:
- The chest print must match PRINT_CROP EXACTLY: shapes/contours, colors, text, placement.
- Do NOT redraw “inspired by” and do NOT replace the print with another.
- If you cannot transfer the print exactly, it is better to keep the person’s original print, but do NOT invent a new one.

OUTPUT:
One photorealistic image: same person/background, wearing TOP_FULL+PRINT_CROP+BOTTOM(+SHOES).
""".strip()

# ---- PASS 2: add ONLY outerwear on top of the already correct outfit ----
TRYON_OUTER_PASS_PROMPT = """
TASK: PHOTO EDITING (NOT generating a new scene).
Edit PERSON_BASE by changing ONLY the outerwear.

STRICT RULES:
- PERSON_BASE is the only base image. Do NOT change the face, identity, age, gender, body shape, pose, background, lighting, camera angle, or framing.
- You may change ONLY the outerwear (OUTER): put it on top of the current outfit.
- You must NOT change the top (t-shirt/polo), the print, the bottom, or the shoes — they are already correct.

INPUTS (by tags):
OUTER      = outerwear (jacket/coat/raincoat).
PERSON_BASE= the person already wearing the outfit (base).

OUTPUT:
One photorealistic image: same person/background, now wearing OUTER on top.
""".strip()

# ---- (kept your old prompt: you can use it for one-pass if needed) ----
TRYON_OUTFIT_PROMPT = """
TASK: PHOTO EDITING (NOT generating a new scene).
Edit PERSON_BASE by changing ONLY the clothing. The result must look like the same photo: same person, same background.

KEY POINTS (STRICT RULES):
- PERSON_BASE is the only base image. Do NOT change the face, identity, age, gender, body shape, pose, background, lighting, camera angle, or framing.
- You may change ONLY the clothing on the person.
- If OUTER is provided — you MUST put OUTER on top of TOP_FULL (outerwear has top priority).
- If SHOES are provided — you MUST replace the shoes with SHOES.
- If OUTER/SHOES are not provided — do NOT add them “from imagination”.

INPUT DESCRIPTION (by tags, NOT by order):
PRINT_CROP = the exact chest print reference (1:1).
TOP_FULL   = catalog photo of the top (t-shirt/polo): color/cut/material.
BOTTOM     = bottom (pants/jeans): color/cut/material.
SHOES      = shoes (if provided).
OUTER      = outerwear (if provided).
PERSON_BASE= the person (base).

PRINT IS CRITICAL (MOST IMPORTANT):
- The chest print must match PRINT_CROP EXACTLY, not “inspired by”.
- Copy the print as-is: same shapes, same colors, same text, no replacing words/letters/symbols.
- If the print cannot be transferred exactly, it is better to keep the person’s original print, but do NOT invent a new one.

CLOTHING LAYER PRIORITY:
1) OUTER (if provided) on top of everything
2) TOP_FULL with PRINT_CROP
3) BOTTOM
4) SHOES

OUTPUT:
One photorealistic image: same person/background, wearing the provided items.
""".strip()