from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from app.core.config import settings
import logging

# 로깅 설정 (옵션)
logging.basicConfig()
# SQLAlchemy 쿼리를 디버그 레벨에서 볼 수 있게 설정합니다.
# logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)

# 1. 비동기 엔진 생성
# config.py에서 정의한 DATABASE_URL(기본값: sqlite:///./test.db)을 사용합니다.
# 비동기 처리를 위해 'sqlite' -> 'sqlite+aiosqlite'로 프로토콜을 변경합니다.
SQLALCHEMY_DATABASE_URL = settings.DATABASE_URL.replace(
    "sqlite:///", "sqlite+aiosqlite:///"
)

# echo=True는 실행되는 SQL 쿼리를 콘솔에 출력하여 디버깅에 도움을 줍니다.
async_engine = create_async_engine(
    SQLALCHEMY_DATABASE_URL, echo=True, future=True  # SQLAlchemy 2.0 스타일 사용
)

# 2. 세션 로컬 생성기
# 데이터베이스와의 상호작용을 위한 세션을 만듭니다.
AsyncSessionLocal = sessionmaker(
    async_engine,
    class_=AsyncSession,
    expire_on_commit=False,  # 세션이 커밋된 후에도 객체가 만료되지 않도록 설정
)

# 3. 모델 정의를 위한 Base 클래스
# 모든 SQLAlchemy 모델이 상속받을 기본 클래스입니다.
Base = declarative_base()


# 4. 의존성 주입(Dependency Injection)을 위한 함수
# FastAPI 라우터에서 이 함수를 호출하여 DB 세션을 얻고, 작업이 끝나면 세션을 닫습니다.
async def get_db_session() -> AsyncSession:
    """
    DB 세션을 생성하고, 작업 후 세션을 닫는 비동기 제너레이터 함수입니다.
    FastAPI의 Depends(get_db_session)으로 사용됩니다.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
