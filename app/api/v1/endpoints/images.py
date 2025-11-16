# app/routers/v1/images.py

from fastapi import APIRouter, File, UploadFile, Form, HTTPException, status, Depends
from fastapi.concurrency import run_in_threadpool
from captioning_module.model import image_captioner
from app.schemas.image import BlipResult

# 🌟 이 파일의 라우터 인스턴스를 생성합니다.
router = APIRouter()


@router.post(
    "/analyze/",
    response_model=BlipResult,
    summary="Step 1: 이미지 분석 및 BLIP 캡션 반환",
)
async def analyze_image_endpoint(image_file: UploadFile = File(...)):
    """
    업로드된 사진 파일을 BLIP 모델로 분석하여 캡션(문자열)만 반환합니다.
    """

    image_data = await image_file.read()

    try:
        # run_in_threadpool을 사용하여 CPU-Bound 작업을 안전하게 실행
        caption = await run_in_threadpool(image_captioner.get_blip_analyze, image_data)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Image analysis failed: {e}",
        )

    if not caption:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Caption generation failed."
        )

    return BlipResult(caption=caption)
