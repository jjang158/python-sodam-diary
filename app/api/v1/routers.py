# app/api/v1/routers.py

from fastapi import APIRouter
from .endpoints import images, captioning
api_router = APIRouter()

# /v1 경로에 images_router를 포함시킵니다.
api_router.include_router(images.router, prefix="/v1", tags=["images"])
api_router.include_router(captioning.router, prefix="/v1", tags=["captioning"])