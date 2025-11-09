# app/routers/api.py

from fastapi import APIRouter
# 🌟 v1/images.py에서 정의한 router를 가져옵니다.
from .v1.images import router as images_router

api_router = APIRouter()

# /v1 경로에 images_router를 포함시킵니다.
api_router.include_router(images_router, prefix="/v1", tags=["v1-Images"]) 

# 필요하다면 다른 버전(v2) 라우터를 여기에 추가할 수 있습니다.