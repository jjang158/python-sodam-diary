from decouple import config
from typing import Optional
import os
import google.generativeai as genai
import openai

# Pydantic BaseSettings를 사용하는 것이 표준이지만, 
# 여기서는 기존 Django 프로젝트의 decouple 사용 패턴을 유지하며 클래스로 설정값을 모읍니다.

class Settings:
    """
    프로젝트의 모든 전역 설정을 중앙 집중적으로 관리하는 클래스입니다.
    """
    # --- LLM API 키 설정 ---
    GEMINI_API_KEY: Optional[str] = config("GEMINI_API_KEY", default=None)
    CHATGPT_API_KEY: Optional[str] = config("CHATGPT_API_KEY", default=None)
    
    # --- 토큰 제한 설정 (기존 views.py에서 가져옴) ---
    # 환경 변수에 없으면 기본값 50000 사용
    DAILY_TOKEN_LIMIT: int = config("DAILY_TOKEN_LIMIT", default=50000, cast=int)
    
    # --- 데이터베이스 설정 (마일스톤 1.3에서 사용 예정) ---
    # 기존 SQLite를 임시로 사용하거나 PostgreSQL 연결 문자열을 준비합니다.
    DATABASE_URL: str = config("DATABASE_URL", default="sqlite:///./test.db")


# 설정 인스턴스 생성
settings = Settings()

# --- LLM 클라이언트 초기화 (초기화 로직도 core에 포함) ---
# 기존 views.py의 초기화 로직을 그대로 옮겨옵니다.
try:
    if settings.GEMINI_API_KEY:
        genai.configure(api_key=settings.GEMINI_API_KEY)
        print("Gemini API configured successfully.")
    else:
        print("Warning: GEMINI_API_KEY not found.")
except Exception as e:
    print(f"Error configuring Gemini API: {e}")

try:
    if settings.CHATGPT_API_KEY:
        openai.api_key = settings.CHATGPT_API_KEY
        print("ChatGPT API configured successfully.")
    else:
        print("Warning: CHATGPT_API_KEY not found.")
except Exception as e:
    print(f"Error configuring ChatGPT API: {e}")

