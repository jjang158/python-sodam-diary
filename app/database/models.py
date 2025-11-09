# app/database/models.py

from sqlalchemy import Column, Integer, String, Text, Date, DateTime, Numeric
from sqlalchemy.sql import func
from app.database.database import Base  # database.py에서 정의한 Base 상속


# --- 1. Image 모델 (기존 Django Image 모델 대체) ---
class ImageModel(Base):
    """
    이미지 캡셔닝 결과를 저장하는 테이블 모델
    """

    __tablename__ = "images"

    id = Column(Integer, primary_key=True, index=True)

    # 텍스트 필드
    file = Column(String(255), nullable=False)
    refined_caption = Column(Text, nullable=False)
    blip_text = Column(Text, nullable=True, default="")
    # clip_text = Column(Text, nullable=True, default="")
    file_info = Column(Text, nullable=True)
    location = Column(String(100), nullable=True)

    # 키워드를 저장할 새 컬럼 추가
    keywords = Column(String, nullable=True) # 콤마로 구분된 문자열 저장 가정

    # 위치 정보 (DecimalField 대체)
    # Django는 DecimalField를 사용했지만, SQLAlchemy는 Numeric 타입을 사용합니다.
    latitude = Column(Numeric(precision=9, scale=6), nullable=True)
    longitude = Column(Numeric(precision=9, scale=6), nullable=True)

    # 생성 시각 (자동 저장)
    created_at = Column(DateTime, default=func.now(), nullable=False)