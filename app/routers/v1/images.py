# app/routers/v1/images.py

from fastapi import APIRouter, File, UploadFile, Form, HTTPException, status, Depends
# **필수 Import 추가:** CPU 바운드 작업을 위해 run_in_threadpool
from fastapi.concurrency import run_in_threadpool 
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, Dict, Any, List # List 추가

# 기존 BLIP 모델 로직 (CLIP 관련 로직은 이미 삭제되었다고 가정)
from captioning_module.model import image_captioner

# 새로 작성한 로직들 (비즈니스 로직 및 DB)
from app.services.llm_service import get_refined_caption_and_keywords_with_chatgpt_async
from app.services import crud 
from app.schemas.image import BlipResult, GenerateRequest, LlmResult, ImageCreate # 🌟 새로운 스키마 import
from app.database.database import get_db_session 

# 🌟 이 파일의 라우터 인스턴스를 생성합니다.
router = APIRouter()

# ❌ 기존의 create_caption 함수는 이 파일에서 삭제됩니다. ❌


# ----------------------------------------------------
# A. Step 1: 사진 분석 API 구현 (POST /analyze/)
# ----------------------------------------------------
@router.post("/analyze/", response_model=BlipResult, summary="Step 1: 이미지 분석 및 BLIP 캡션 반환")
async def analyze_image_endpoint(image_file: UploadFile = File(...)):
    """
    업로드된 사진 파일을 BLIP 모델로 분석하여 캡션(문자열)만 반환합니다.
    """
    
    image_data = await image_file.read()
    
    try:
        # run_in_threadpool을 사용하여 CPU-Bound 작업을 안전하게 실행
        caption = await run_in_threadpool(image_captioner.get_blip_analyze, image_data)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Image analysis failed: {e}")

    if not caption:
         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Caption generation failed.")
        
    return BlipResult(caption=caption)


# ----------------------------------------------------
# B. Step 2: LLM 해설 및 태그 생성 API 구현 (POST /generate/)
# ----------------------------------------------------
@router.post("/generate/", response_model=LlmResult, summary="Step 2: 사용자 입력과 BLIP 결과를 기반으로 LLM 일기/태그 생성")
async def generate_llm_result(
    request: GenerateRequest,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Step 1의 캡션과 사용자의 추가 정보를 받아 LLM을 호출하여 최종 일기 해설과 단어 태그를 생성하고 DB에 저장합니다.
    """
    
    # 1. LLM 프롬프트 구성
    full_prompt = (
        f"당신은 사용자의 사진과 생각을 바탕으로 일기를 작성해주는 인공지능입니다.\n"
        f"다음 정보를 바탕으로 일기 해설('diary')과 핵심 단어 태그('tags')를 JSON 형식으로 생성하세요:\n"
        f"사용자 입력 정보: {request.user_input}\n"
        f"사진으로부터 추출된 설명: {request.blip_caption}"
    )

    # 2. LLM 서비스 호출
    try:
        llm_result = await get_refined_caption_and_keywords_with_chatgpt_async(
            full_prompt, request.user_input
        )
        
        refined_caption = llm_result.get("refined_caption", "LLM 결과 추출 오류")
        keywords = llm_result.get("keywords", [])
        
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"LLM generation failed: {e}")
        
    
    # 3. DB 저장을 위한 Pydantic 데이터 준비
    data_to_create = ImageCreate(
        file="BLIP_LLM_Processed", # 임시 파일명
        refined_caption=refined_caption,
        blip_text=request.blip_caption,
        keywords=",".join(keywords) if keywords else None,
        file_info=request.user_input, 
        latitude=request.latitude,
        longitude=request.longitude,
        location=request.location,
    )

    # 4. DB 저장 및 응답 반환
    try:
        saved_image = await crud.create_image_data(db, data_to_create)

        # LlmResult 스키마에 맞춰 최종 응답 반환
        return LlmResult(diary=saved_image.refined_caption, tags=keywords) 

    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"데이터베이스 저장 중 오류 발생: {e}",
        )