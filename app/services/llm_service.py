# app/services/llm_service.py

import openai
import google.generativeai as genai
from typing import Dict, Any
from app.core.config import settings
from captioning_module.model import image_captioner  # 모델 로직 재사용
import time  # 토큰 사용량 계산 및 출력을 위해 사용
from openai import OpenAI, AsyncOpenAI # AsyncOpenAI를 임포트합니다.
import json  # JSON 응답 파싱을 위해 사용

if settings.CHATGPT_API_KEY:
    async_openai_client = AsyncOpenAI(api_key=settings.CHATGPT_API_KEY)
else:
    async_openai_client = None


# --- 프롬프트 생성 함수 (기존 로직 유지) ---
def set_prompt(original_caption: str, file_info: str) -> str:
    """LLM에 전달할 사용자 프롬프트를 생성합니다."""
    return (
        f"당신은 시각 장애인 친구에게 사진을 설명해주는 다정하고 친근한 친구입니다.\n"
        f"이 사진의 원래 캡션은 '{original_caption}'입니다.\n"
        f"친구의 추가 설명은 '{file_info}'입니다.\n"
        f"이 두 정보를 바탕으로, 친구가 눈으로 보는 것처럼 사진의 상황, 분위기, 그리고 느껴지는 감정들을 생생하고 직관적인 언어로 전달해주세요."
    )


def set_test_prompt(original_caption: str, file_info: str) -> str:
    """ChatGPT용 사용자 프롬프트 (시스템 프롬프트와 조합)"""
    return f"""Photo description: {original_caption},
            Additional info: {file_info}"""


# --- 토큰 사용량 체크 로직 (Persistence 제거, Limit 체크는 유지) ---
def get_estimated_tokens(text: str, is_korean: bool = True) -> int:
    """텍스트 길이에 따라 토큰을 추정합니다."""
    # 한국어는 문자당 토큰 비율이 높음 (약 1.5~2배)
    ratio = 2 if is_korean else 1.2
    return int(len(text.split()) * ratio)


# **참고**: 토큰 사용량의 DB 영속성(저장)이 제거되었으므로,
# 일일 제한(DAILY_TOKEN_LIMIT) 체크는 '현재 호출의 예상 비용'만 체크하는 방식으로 단순화됩니다.
# 실제 배포 시에는 Redis 등으로 일일 사용량을 다시 관리해야 합니다.


# --- LLM 연동 및 캡션 생성 함수 (Gemini) ---
async def get_refined_caption_with_gemini_async(original_caption: str, file_info: str):
    if not settings.GEMINI_API_KEY:
        return "LLM API 호출 실패: Gemini API key is not configured."

    prompt = set_prompt(original_caption, file_info)

    # 토큰 사용량 체크 (현재 호출의 예상 비용만 확인)
    estimated_input_tokens = get_estimated_tokens(prompt)
    if estimated_input_tokens * 2 >= settings.DAILY_TOKEN_LIMIT:
        return "일일 토큰 사용량 제한에 도달했습니다. (단일 요청 크기 초과)"

    try:
        # NOTE: genai.GenerativeModel.generate_content는 동기 함수입니다.
        # FastAPI에서 동기 함수를 호출하면 다른 요청을 블로킹할 수 있으므로,
        # 실제 환경에서는 run_in_threadpool 등을 사용해야 합니다.
        # 여기서는 로직 포팅을 위해 동기 호출로 둡니다.
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        refined_caption = response.text

        # 토큰 사용량 시뮬레이션 출력 (Persistence는 없음)
        output_tokens = get_estimated_tokens(refined_caption)
        print(
            f"[Gemini] 예상 토큰 사용량 (입력/출력): {estimated_input_tokens}/{output_tokens}. (Limit: {settings.DAILY_TOKEN_LIMIT})"
        )

        return refined_caption
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return f"LLM API 호출 실패: {e}"


async def get_refined_caption_with_chatgpt_async(
    original_caption: str, file_info: str
) -> str:
    """
    ChatGPT API를 사용하여 캡션을 개선하고 응답을 파싱합니다.
    """
    if not async_openai_client:
        return "LLM API 호출 실패: ChatGPT API 키가 설정되지 않았습니다."

    # --- 1. JSON 응답을 위한 시스템 프롬프트 정의 ---
    system_prompt = (
        "You are a helpful assistant that refines an image caption based on provided context. "
        "Your final response MUST be a single JSON object with the key 'refined_caption'. "
        "The value of 'refined_caption' should be the final, refined caption in Korean."
    )
    
    # --- 2. 사용자 입력 프롬프트 생성 ---
    prompt = set_prompt(original_caption, file_info)
    model_name = "gpt-3.5-turbo" # 사용할 모델

    try:
        completion = await async_openai_client.chat.completions.create(
            model=model_name,
            messages=[
                # 시스템 메시지 추가
                {"role": "system", "content": system_prompt}, 
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            response_format={"type": "json_object"}
        )

        # 3. 응답에서 텍스트 추출 및 파싱
        response_text = completion.choices[0].message.content
        data = json.loads(response_text)
        refined_caption = data.get("refined_caption", "캡션 생성 결과 없음")

        return refined_caption

    except Exception as e:
        print(f"Error calling ChatGPT API: {e}") 
        # 이전 오류 메시지 대신 실제 예외를 출력하도록 변경했습니다.
        # 기존에는 'object ChatCompletion can't be used in 'await' expression'
        return "LLM API 호출 실패: ChatGPT API 통신 중 오류 발생"
