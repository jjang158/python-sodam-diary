# app/routers/captioning.py

from fastapi import APIRouter, File, UploadFile, Form, HTTPException, status, Depends
from app.services.llm_service import get_refined_caption_and_keywords_with_chatgpt_async
from app.schemas.image import (
    GenerateRequest,
    LlmResult,
)

router = APIRouter()


@router.post(
    "/generate/",
    response_model=LlmResult,
    summary="Step 2: 사용자 입력과 BLIP 결과를 기반으로 LLM 일기/태그 생성",
)
async def generate_llm_result(
    request: GenerateRequest,
):
    """
    Step 1의 캡션과 사용자의 추가 정보를 받아 LLM을 호출하여 최종 일기 해설과 단어 태그를 생성하고 DB에 저장합니다.
    """

    try:
        llm_result = await get_refined_caption_and_keywords_with_chatgpt_async(
            request.blip_caption, request.user_input
        )
        refined_caption = llm_result.get("refined_caption", "LLM 결과 추출 오류")
        keywords = llm_result.get("keywords", [])

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"LLM generation failed: {e}",
        )

    return LlmResult(diary=refined_caption, tags=keywords)
