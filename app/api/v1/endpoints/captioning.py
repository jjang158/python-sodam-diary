# app/routers/captioning.py

from fastapi import APIRouter, File, UploadFile, Form, HTTPException, status, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, Dict, Any

# 기존 BLIP/CLIP 모델 로직 (동기)
from captioning_module.model import image_captioner

# 새로 작성한 로직들 (비즈니스 로직 및 DB)
from app.services.llm_service import get_refined_caption_and_keywords_with_chatgpt_async
from app.services import crud  # crud.py에서 정의한 DB 상호작용 함수
from app.schemas.image import ImageCreate, Image  # DB 저장용 스키마, 응답용 스키마
from app.database.database import get_db_session  # DB 세션 DI 함수

# NOTE: FastAPI에서 동기 I/O(예: BLIP/CLIP 모델 추론)가 전체 서버를 블로킹하는 것을 방지하기 위해
# from fastapi.concurrency import run_in_threadpool 등을 사용하여 모델 추론을 감싸는 것이 좋습니다.
# 여기서는 코드를 간결하게 유지하기 위해 직접 호출하고, 실제 main.py에서 run_in_threadpool을 사용하는 것을 권장합니다.
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
    # db: AsyncSession = Depends(get_db_session),
):
    """
    Step 1의 캡션과 사용자의 추가 정보를 받아 LLM을 호출하여 최종 일기 해설과 단어 태그를 생성하고 DB에 저장합니다.
    """

    # # 1. LLM 프롬프트 구성
    # full_prompt = (
    #     f"당신은 사용자의 사진과 생각을 바탕으로 일기를 작성해주는 인공지능입니다.\n"
    #     f"다음 정보를 바탕으로 일기 해설('diary')과 핵심 단어 태그('tags')를 JSON 형식으로 생성하세요:\n"
    #     f"사용자 입력 정보: {request.user_input}\n"
    #     f"사진으로부터 추출된 설명: {request.blip_caption}"
    # )

    # 2. LLM 서비스 호출
    try:
        # 🌟 수정: BLIP 캡션(request.blip_caption)과 사용자 입력(request.user_input)만 전달
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


# @router.post(
#     "/images/caption/",
#     response_model=Image,  # 응답 데이터 형식은 Pydantic Image 스키마를 따릅니다.
#     status_code=status.HTTP_201_CREATED,
# )
# async def create_caption(
#     # 1. 클라이언트 요청 데이터 (FastAPI의 Form/UploadFile 처리)
#     file: UploadFile = File(..., description="이미지 파일"),
#     file_info: str = Form("사용자 음성 없음", description="사용자의 음성 입력 텍스트"),
#     latitude: Optional[float] = Form(None, description="위도"),
#     longitude: Optional[float] = Form(None, description="경도"),
#     location: Optional[str] = Form(None, description="위치 정보"),
#     # 2. DB 세션 의존성 주입 (FastAPI Dependency Injection)
#     db: AsyncSession = Depends(get_db_session),
# ):
#     """
#     이미지와 사용자 입력을 받아 캡션을 생성하고 저장하는 엔드포인트입니다.
#     """

#     llm_choice = "chatgpt"  # 기존 views.py의 선택 유지

#     # 1단계: 파일 읽기 및 BLIP/CLIP 분석 (I/O 작업)
#     try:
#         # 비동기로 파일을 읽음
#         image_data = await file.read()

#         # BLIP/CLIP 모델 추론 (동기 함수)
#         analysis_result = image_captioner.analyze_image(image_data)

#         blip_text = analysis_result.get("file_description", "캡션 생성 실패")
#         clip_moods = analysis_result.get("file_moods", [])

#         # 상위 무드를 문자열로 조합
#         clip_text = (
#             ", ".join([mood["label"] for mood in clip_moods])
#             if clip_moods
#             else "분위기 분석 실패"
#         )

#     except Exception as e:
#         print(f"Error analyzing file: {e}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"파일 분석 또는 모델 추론 중 오류 발생: {e}",
#         )

#     # 2단계: LLM을 통한 캡션 개선 (네트워크 I/O 작업)
#     refined_caption = None  # 초기화
#     keywords = []  # 키워드 변수 초기화

#     if llm_choice == "gemini":
#         # Gemini 호출 (미구현 상태)
#         raise HTTPException(
#             status_code=status.HTTP_501_NOT_IMPLEMENTED,
#             detail="Gemini LLM 통합은 아직 구현되지 않았습니다.",
#         )
#     elif llm_choice == "chatgpt":
#         # ChatGPT 호출 및 딕셔너리 응답 처리
#         llm_result = await get_refined_caption_and_keywords_with_chatgpt_async(
#             f"Photo description: {blip_text}, Predicted mood: {clip_text}",
#             file_info,
#         )

#         # 딕셔너리에서 값 추출 (핵심 수정)
#         refined_caption = llm_result.get("refined_caption", "LLM 결과 추출 오류")
#         keywords = llm_result.get("keywords", [])  # 키워드 추출

#     else:
#         raise HTTPException(
#             status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid LLM choice."
#         )

#     # LLM 오류 처리 (refined_caption이 문자열이라고 가정하고 체크)
#     if "LLM API 호출 실패" in refined_caption:
#         raise HTTPException(
#             status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=refined_caption
#         )

#     # 3단계: DB 저장을 위한 Pydantic 데이터 준비
#     # DB에 저장할 때 필요한 데이터만 ImageCreate 스키마에 맞춥니다.
#     data_to_create = ImageCreate(
#         file=file.filename,  # UploadFile에서 파일명 추출
#         refined_caption=refined_caption,
#         blip_text=blip_text,
#         clip_text=clip_text,
#         # **추가된 필드:** DB에 저장할 키워드 리스트를 문자열로 변환 (DB 설계에 따라 다름)
#         # 예: 키워드 리스트를 콤마로 구분된 문자열로 저장한다고 가정
#         keywords=",".join(keywords) if keywords else None,
#         file_info=file_info,
#         latitude=latitude,
#         longitude=longitude,
#         location=location,
#     )

#     # 4단계: DB 저장 (비동기)
#     try:
#         # crud.py에서 정의한 함수를 사용하여 DB에 저장
#         saved_image = await crud.create_image_data(db, data_to_create)

#         # 5단계: 응답 반환
#         return saved_image  # 응답 모델 Image(id, created_at 포함)를 반환

#     except Exception as e:
#         print(f"Database saving error: {e}")
#         await db.rollback()
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"데이터베이스 저장 중 오류 발생: {e}",
#         )
