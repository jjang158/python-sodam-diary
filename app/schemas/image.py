# app/schemas/image.py

from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
from decimal import Decimal


# ----------------------------------------------------------------------
# A. Step 1: 사진 분석 API (/analyze/) 응답 스키마
# ----------------------------------------------------------------------

# NOTE: Step 1 요청은 UploadFile을 사용하므로 별도의 Pydantic 요청 스키마가 필요 없습니다.


class BlipResult(BaseModel):
    """
    Step 1 응답 스키마: /analyze/ 엔드포인트의 응답 (BLIP 캡션)
    """

    caption: str


# ----------------------------------------------------------------------
# B. Step 2: LLM 해설 생성 API (/generate/) 요청/응답 스키마
# ----------------------------------------------------------------------


class GenerateRequest(BaseModel):
    """
    Step 2 요청 스키마: /generate/ 엔드포인트에 전달되는 데이터 정의
    """

    user_input: Optional[str] = None  # 사용자가 사용자의 음성/텍스트 입력
    blip_caption: str  # Step 1에서 받은 BLIP 분석 결과
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    location: Optional[str] = None


class LlmResult(BaseModel):
    """
    Step 2 응답 스키마: LLM의 최종 결과 (일기 해설과 태그 리스트)
    """

    diary: str
    tags: List[str]


# ----------------------------------------------------------------------
# C. 데이터베이스 엔티티(DB Entity) 스키마 정의 수정 (CLIP_TEXT 제거)
# ----------------------------------------------------------------------


class ImageBase(BaseModel):
    """
    Image 테이블의 기본 필드 (생성 시 필요)
    'clip_text' 필드가 제거되었습니다.
    """

    file: str
    refined_caption: str  # LLM이 생성한 최종 일기 해설 (DB 저장용)
    blip_text: str  # BLIP 결과 (DB 저장용)
    # ❌ clip_text 필드 삭제 ❌
    file_info: Optional[str] = None  # 이전 버전과의 호환성을 위해 유지 (사용자 입력)
    location: Optional[str] = None
    # DecimalField 대신 float 사용 (Python-DB 간 호환성 고려)
    latitude: Optional[Decimal] = None
    longitude: Optional[Decimal] = None
    # 🌟 핵심 필드: DB에 저장할 키워드 문자열 필드
    keywords: Optional[str] = None


class ImageCreate(ImageBase):
    """
    데이터를 생성(INSERT)할 때 사용되는 스키마
    """

    pass


class Image(ImageBase):
    """
    데이터베이스에서 읽어올 때 사용되는 스키마 (응답 시에도 사용)
    """

    id: int
    created_at: datetime

    class Config:
        # Pydantic 모델을 ORM 객체와 호환되게 설정
        from_attributes = True

        # Decimal 타입을 JSON으로 직렬화할 때 float로 변환되도록 설정
        json_encoders = {
            Decimal: float,
        }
