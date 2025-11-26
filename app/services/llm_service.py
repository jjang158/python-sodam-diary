# app/services/llm_service.py

# import openai
# import google.generativeai as genai
from typing import Dict, Any
from app.core.config import settings
from captioning_module.model import image_captioner  # 모델 로직 재사용
import time  # 토큰 사용량 계산 및 출력을 위해 사용
from openai import AsyncOpenAI  # AsyncOpenAI를 임포트합니다.
import json  # JSON 응답 파싱을 위해 사용

if settings.CHATGPT_API_KEY:
    async_openai_client = AsyncOpenAI(api_key=settings.CHATGPT_API_KEY)
else:
    async_openai_client = None


# --- 토큰 사용량 체크 로직 (Persistence 제거, Limit 체크는 유지) ---
def get_estimated_tokens(text: str, is_korean: bool = True) -> int:
    """텍스트 길이에 따라 토큰을 추정합니다."""
    # 한국어는 문자당 토큰 비율이 높음 (약 1.5~2배)
    ratio = 2 if is_korean else 1.2
    return int(len(text.split()) * ratio)


# --- 2. 유저 프롬프트 (입력 + 요청만) ---
def set_prompt_for_keyword(image_caption: str, user_diary_summary: str) -> str:
    return (

        f"## 입력 정보\n"
        f"1. 사진 정보: {image_caption}\n"
        f"2. 사용자 음성 일기: {user_diary_summary}\n\n"
        f"## 작성 규칙\n"
        f"### 구조 (5문장 이상)\n"
        f"- 시작: 사진 속 장면으로 자연스럽게 시작\n"
        f"- 중간: 사용자의 감정과 경험을 녹여냄\n"
        f"### 스타일\n"
        f"- 1인칭 과거형 일기체 (예: ~했다, ~였다, ~이었다)\n"
        f"- 개인이 자신의 경험을 회상하며 쓰는 톤\n"
        f"- '-습니다/-어요/-네요' 같은 설명체, 경어체 절대 금지\n"
        f"- TTS 고려하여 쉼표, 마침표 적절히 배치\n"
        f"- 솔직한 톤\n\n"
        f"### 비중\n"
        f"- 사진 정보 30%, 사용자 음성 일기 70%\n"
        f"- 사용자 음성 일기 필수 반영\n\n"
        f"### 금지사항\n"
        f"- '사진에는~', '캡션에는~' 같은 메타적 표현 금지\n"
        f"- 사진에 없는 정보 창작 금지\n"
        f"- 사용자 입력의 핵심 의미 왜곡/누락 금지\n\n"
        f"## 출력 형식\n"
        f"아래 JSON 형식으로만 출력하세요. 다른 설명 금지.\n"
        f"```json\n"
        f"{{\n"
        f'  "caption": "여기에 작성 지침에 따른 1인칭 해설 전체 내용을 작성합니다.",\n'
        f'  "keywords": "키워드1, 키워드2, 키워드3, ... (최대 10개)"\n'
        f"}}\n"
        f"```"
    )


async def get_refined_caption_and_keywords_with_chatgpt_async(
    original_caption: str, file_info: str
) -> Dict[str, Any]:  # 응답 타입을 Dict로 변경
    """
    ChatGPT API를 사용하여 캡션 개선 및 10개 키워드를 JSON으로 받아 파싱합니다.
    """
    if not async_openai_client:
        # 키가 설정되지 않은 경우에도 딕셔너리 형태로 반환
        return {
            "refined_caption": "LLM API 호출 실패: ChatGPT API 키가 설정되지 않았습니다.",
            "keywords": [],
        }

    # --- 1. 시스템 프롬프트 (역할 + 형식만) ---
    system_prompt = (
        "당신은 시각장애인을 위한 사진 일기 앱을 사용하는 시각장애인 입니다."
        "사진 정보와 사용자의 음성 일기를 자연스럽게 결합하여 TTS로 읽힐 완성된 일기를 작성하세요."
        "You must always respond in valid JSON format with exactly two keys: "
        "'refined_caption' (string, in Korean) and 'keywords' (array of exactly 10 Korean nouns or noun phrases). "
        "but if you cannot find enough keywords, return as many as you can. "
        "Do not include any extra text outside the JSON."
    )

    # --- 2. 사용자 입력 프롬프트 생성 (새로운 함수 사용) ---
    prompt = set_prompt_for_keyword(original_caption, file_info)
    model_name = "gpt-3.5-turbo"  # 사용할 모델

    try:
        completion = await async_openai_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,  # 형식 일관성 ↑
            response_format={"type": "json_object"},
        )

        # 3. 응답에서 텍스트 추출 및 파싱
        response_text = completion.choices[0].message.content
        data = json.loads(response_text)

        # 4. 키와 캡션 추출
        refined_caption = data.get("refined_caption", "캡션 생성 결과 없음")
        keywords = data.get("keywords", [])  # 키워드 리스트 추출

        # 최종 반환: 딕셔너리 형태로 캡션과 키워드 모두 반환
        return {"refined_caption": refined_caption, "keywords": keywords}

    except Exception as e:
        print(f"Error calling ChatGPT API: {e}")
        return {"refined_caption": f"LLM API 호출 실패: {e}", "keywords": []}


# 🌟 전역 클라이언트를 사용하거나, 설정되지 않았다면 None을 반환하도록 수정
async def translate_to_korean_async(english_text: str) -> str:
    """
    GPT를 사용하여 영어 텍스트를 한국어로 번역합니다.
    (경량 프롬프트로 토큰 사용 최소화)
    """
    # 🌟 전역으로 생성된 클라이언트(async_openai_client)를 사용합니다.
    client = async_openai_client

    if not client:
        # LLM 키가 설정되지 않았다면 번역 불가능
        print("LLM Translation skipped: ChatGPT API key is not configured.")
        return english_text

    system_prompt = "You are a professional Korean translator. Translate the given text into natural Korean. Do not add any explanations or extra text."

    try:
        response = await client.chat.completions.create(  # client(전역) 사용
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": english_text},
            ],
            temperature=0.1,
            max_tokens=200,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"LLM Translation failed: {e}")
        return english_text
