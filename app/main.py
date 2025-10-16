from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.routers import captioning
from app.database.database import async_engine, Base
from app.database.models import *  # 모델을 import해야 Base.metadata가 테이블을 인식


# --- 1. DB 초기화 컨텍스트 관리자 ---
async def create_db_tables():
    """
    SQLAlchemy 엔진을 사용하여 DB 테이블이 없으면 생성합니다.
    """
    async with async_engine.begin() as conn:
        # 테이블 존재 여부에 관계없이 Base.metadata에 정의된 모든 테이블을 생성합니다.
        # (이미 존재하면 건너뜁니다.)
        await conn.run_sync(Base.metadata.create_all)
    print("Database tables initialized successfully.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 서버 시작 및 종료 시 실행할 작업을 정의합니다.
    """
    # 서버 시작 시 (Startup)
    await create_db_tables()
    yield
    # 서버 종료 시 (Shutdown)
    # (필요한 경우 여기에 정리 코드를 추가합니다.)


# --- 2. FastAPI 인스턴스 생성 ---
app = FastAPI(
    title="Sodam Diary API",
    description="Visually Impaired Captioning Server with BLIP/CLIP and LLM",
    version="1.0.0",
    lifespan=lifespan,  # 라이프스팬 매니저 적용
)

# --- 3. 라우터 등록 ---
app.include_router(captioning.router, prefix="/api/v1")


@app.get("/")
def read_root():
    return {"Hello": "Sodam Diary FastAPI Server is running!"}


# --- 4. 서버 실행 가이드 ---
# 로컬에서 서버를 실행하려면 터미널에서 다음 명령을 사용해야 합니다.
# uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
