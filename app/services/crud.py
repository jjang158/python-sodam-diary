from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.database.models import ImageModel
from app.schemas.image import (
    ImageCreate,
    Image,
)  # ImageCreate: 입력 데이터, Image: 출력 데이터


# --- 1. 데이터 생성(CREATE) ---
async def create_image_data(db: AsyncSession, image_data: ImageCreate) -> Image:
    """
    캡셔닝된 이미지 데이터를 데이터베이스에 저장합니다.

    Args:
        db: SQLAlchemy 비동기 세션.
        image_data: Pydantic ImageCreate 스키마를 따르는 데이터.

    Returns:
        DB에 저장된 Image 객체(Pydantic Image 스키마)
    """
    # Pydantic 모델을 딕셔너리로 변환하여 SQLAlchemy 모델 객체 생성
    db_image = ImageModel(**image_data.model_dump())

    # DB 세션에 추가
    db.add(db_image)

    # DB에 커밋 (비동기)
    await db.commit()

    # DB에서 최신 데이터(id, created_at 포함)를 반영하도록 새로고침
    await db.refresh(db_image)

    # DB 모델 객체(db_image)를 Pydantic Image 스키마로 변환하여 반환
    return Image.model_validate(db_image)


# --- 2. 데이터 조회(READ) ---
async def get_image_data(db: AsyncSession, image_id: int) -> ImageModel | None:
    """
    ID를 기준으로 단일 이미지 데이터를 조회합니다.

    Args:
        db: SQLAlchemy 비동기 세션.
        image_id: 조회할 이미지의 ID.

    Returns:
        ImageModel 객체 또는 None.
    """
    # SQLAlchemy 2.0 스타일의 비동기 select 문 사용
    stmt = select(ImageModel).where(ImageModel.id == image_id)

    # DB에서 실행 (await 필수)
    result = await db.execute(stmt)

    # 결과 중 첫 번째 row를 가져옵니다.
    db_image = result.scalars().first()

    return db_image


# (필요하다면, 모든 이미지 조회, 업데이트, 삭제 함수 등을 여기에 추가합니다.)
