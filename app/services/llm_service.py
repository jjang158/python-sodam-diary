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


# --- 프롬프트 생성 함수 수정 ---
def set_prompt_for_keyword(original_caption: str, file_info: str) -> str:
    """LLM에 전달할 사용자 프롬프트 및 키워드 추출 지시를 생성합니다."""
    return (
        # 🌟 통합: 구체적인 페르소나 정의
        f"당신은 시각 장애인 친구에게 사진을 설명해주는 다정하고 친근한 도우미입니다.\n\n" 
        
        # 입력 데이터 및 중요성 강조
        f"제공된 '사용자 추가 정보'에는 사진 속 인물의 이름이나 중요 정보가 포함될 수 있습니다. 최종 해설 작성 시, 대명사 대신 '사용자 추가 정보'에 포함된 구체적인 이름이나 정보를 반드시 사용하여 서술해 주세요.\n\n"
        
        f"[입력 데이터]\n"
        f"1. 사진 캡션: '{original_caption}'\n"
        f"2. 사용자 추가 정보: '{file_info}'\n\n"
        
        # 🌟 명령 명확화: 이제 A, B 대신 '해설'과 '키워드'로 명확히 지시
        f"[요청 사항]\n"
        f"1. 해설 생성: 위에 제시된 정보를 바탕으로, 눈으로 보는 것처럼 사진의 상황, 분위기, 감정을 생생하고 직관적인 언어로 전달하는 최종 해설을 작성해주세요.\n"
        f"2. 키워드 추출: 이 사진과 해설을 대표하는 **객체, 장소, 분위기, 감정, 인물의 이름**을 포함하는 **10개의 핵심 키워드**를 추출해주세요. 키워드는 명사 또는 명사구 형태여야 합니다."
    )a

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

    # --- 1. JSON 응답을 위한 시스템 프롬프트 정의 (키워드 항목 추가) ---
    system_prompt = (
        "You are a helpful assistant that refines an image caption based on provided context. "
        "Your final response MUST be a single JSON object with two keys: 'refined_caption' (String) and 'keywords' (Array of Strings). "
        "The value of 'refined_caption' should be the final, refined caption in Korean. "
        "The value of 'keywords' MUST be an array containing exactly 10 keywords in Korean. "
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
            temperature=0.7,
            response_format={"type": "json_object"},  # JSON 형식 요청
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
