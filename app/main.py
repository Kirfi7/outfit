from fastapi import FastAPI

try:
    from app.api.routes import health_router, sizes_router, tryon_router
    from app.core.config import get_settings
except ModuleNotFoundError:
    # fallback for runtime where cwd is app/ and uvicorn target is main:app
    from api.routes import health_router, sizes_router, tryon_router
    from core.config import get_settings


def create_app() -> FastAPI:
    # validate settings/keys at startup (preserve fail-fast behavior)
    get_settings()

    application = FastAPI()
    application.include_router(health_router)
    application.include_router(sizes_router)
    application.include_router(tryon_router)
    return application


app = create_app()
