# app/schemas/image.py

from pydantic import BaseModel
from typing import Optional
from datetime import datetime, date
from decimal import Decimal


# --- 1. 요청(Request) 스키마 정의 ---
# FastAPI 라우터가 클라이언트로부터 받는 데이터를 정의합니다.
# 파일 자체는 UploadFile 형태로 받으므로, form-data 필드만 정의합니다.
class CaptioningRequest(BaseModel):
    """
    POST /api/v1/images/caption/ 엔드포인트에 전달되는 form-data 필드 스키마
    FastAPI는 File과 Form으로 받지만, 데이터 형태를 Pydantic으로 정의하는 것이 좋습니다.
    """

    file_info: str = "사용자 음성 없음"  # Form("사용자 음성 없음") 대체
    latitude: Optional[float] = None  # Form(None) 대체
    longitude: Optional[float] = None  # Form(None) 대체
    location: Optional[str] = None  # Form(None) 대체

    # 이 모델은 UploadFile 객체를 직접 다루지 않고, 메타데이터 필드만 정의합니다.
    # 실제 라우터에서는 FastAPI의 Form()과 UploadFile()을 조합하여 사용합니다.


# --- 2. 데이터베이스 엔티티(DB Entity) 스키마 정의 ---
# 데이터베이스에 저장될 데이터의 형태를 정의합니다. (기존 Image 모델 대체)
class ImageBase(BaseModel):
    """
    Image 테이블의 기본 필드 (생성 시 필요)
    """

    file: str
    refined_caption: str
    blip_text: str
    clip_text: str
    file_info: Optional[str] = None
    location: Optional[str] = None
    # DecimalField 대신 float 사용 (Python-DB 간 호환성 고려)
    latitude: Optional[Decimal] = None
    longitude: Optional[Decimal] = None
    # 🌟 핵심 수정: DB에 저장할 키워드 문자열 필드를 Base에 추가
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

    id: int  # Django의 auto increment ID
    created_at: datetime

    class Config:
        # FastAPI가 Pydantic 모델을 ORM 객체와 호환되게 설정합니다.
        # (예: id=1 대신 'id'=1 처럼 딕셔너리 키로 접근 가능)
        from_attributes = True

        # Decimal 타입을 JSON으로 직렬화할 때 float로 변환되도록 설정
        json_encoders = {
            Decimal: float,
        }
