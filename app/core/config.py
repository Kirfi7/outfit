from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    api_txt_path: str = Field(default="/home/admin/api.txt", alias="API_TXT_PATH")

    aitunnel_url: str = Field(default="https://api.aitunnel.ru/v1/chat/completions", alias="AITUNNEL_URL")
    aitunnel_model: str = Field(default="gemini-2.5-flash-lite", alias="AITUNNEL_MODEL")
    aitunnel_key: str = Field(default="", alias="AITUNNEL_KEY")

    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    # gemini_model: str = Field(default="gemini-2.5-flash-image", alias="GEMINI_MODEL")
    gemini_model: str = Field(default="gemini-3-pro-image-preview", alias="GEMINI_MODEL")
    image_ar: str = Field(default="9:16", alias="IMAGE_AR")
    image_size: str = Field(default="1K", alias="IMAGE_SIZE")
    proxy_url: str = Field(default="http://sve2Zh:1W4P9D@138.219.173.226:8000", alias="PROXY_URL")


def load_keys_from_txt(path: str) -> None:
    p = Path(path)
    if not p.exists():
        raise RuntimeError(f"api.txt not found at: {path}")

    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value:
            os.environ[key] = value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    temp = Settings()
    load_keys_from_txt(temp.api_txt_path)
    settings = Settings()

    if not settings.aitunnel_key.strip():
        raise RuntimeError("AITUNNEL_KEY is missing in configured api.txt/.env")
    if not settings.gemini_api_key.strip():
        raise RuntimeError("GEMINI_API_KEY is missing in configured api.txt/.env")

    return settings
