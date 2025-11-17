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
        f"당신은 시각장애인을 위한 사진 일기 앱의 스토리텔러입니다.\n"
        f"제공된 사진 캡셔닝 데이터와 사용자의 음성 일기 요약을 자연스럽게 결합하여, 듣기 편하고 감성적인 하나의 **완성된 일기 스크립트**를 작성하되, 사용자 정보를 최대한 반영하세요.\n\n"
        f"## 입력 정보\n"
        f"1. **사진 캡셔닝 데이터 (시각적 정보)**: {image_caption}\n"
        f"2. **사용자 음성 일기 요약 (감정, 경험)**: {user_diary_summary}\n\n"
        f"# 작성 가이드라인\n"
        f"## 1. 구조 및 흐름\n"
        f"- **시작**: 사진 설명으로 자연스럽게 시작하여 장면을 그려주세요.\n"
        f"- **중간**: 사용자의 감정, 생각, 경험을 중간에 자연스럽게 녹여내세요.\n"
        f"- **마무리**: 따뜻하고 회상적인 톤으로 마무리하세요.\n"
        f"## 2. 스타일\n"
        f"- **시점**: 1인칭으로 **사용자 입력 톤에 맞춰** 가장 자연스러운 것을 선택하세요.\n"
        f"- **문체**: **구어체**로 자연스럽고 친근하게 작성하세요 (TTS로 읽어줄 것을 고려).\n"
        f"- **길이**: 사용자 입력 양에 따라 가변적이며, 자연스러운 문단 구성 (2-3문단 권장)을 지향하세요.\n"
        f"- **톤**: 따뜻하고 공감적이며, 지나치게 감상적이지 않은 톤을 유지하세요.\n"
        f"## 3. 필수 포함 요소\n"
        f"- 사진의 주요 시각적 요소\n"
        f"- 사용자가 언급한 핵심 감정이나 사건\n"
        f"- 시간적/공간적 배경 (사용자 입력에 있는 경우)\n"
        f"## 4. 주의사항\n"
        f"- 사진에 없는 정보를 과도하게 추론하거나 창작하지 마세요.\n"
        f"- 사용자 입력의 핵심 의미를 왜곡하거나 정보를 누락하지 마세요.\n"
        f"- 사진 설명과 사용자 이야기가 자연스럽게 연결되도록 하세요.\n"
        f"- **'사진에는~', '캡션 데이터에는~'처럼 메타적 표현은 피하고 직접적으로 묘사**하세요.\n"
        f"- **TTS로 읽힐 것을 고려하여 쉼표와 마침표를 적절히 배치**하세요.\n"
        f"## 출력 형식 (Response Format)\n"
        f"요청한 정보를 다음 Python 딕셔너리 형식에 맞춰서 JSON으로 출력하세요. 다른 설명이나 텍스트는 일체 포함하지 마세요.\n"
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
        "You are a storyteller who describes photos emotionally. "
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
