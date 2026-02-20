from typing import Optional

from pydantic import BaseModel


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
