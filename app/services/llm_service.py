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
def set_prompt_for_keyword(original_caption: str, file_info: str) -> str:
    return (
        f"당신은 사용자가 제출한 사진 캡션과 추가 정보를 기반으로 **일기 형식의 해설(Caption)** 및 **키워드(Keywords)**를 생성하는 AI입니다.\n\n"
        
        f"## 입력 정보\n"
        f"1. **사진 캡션 (BLIP/CLIP)**: {original_caption}\n"
        f"2. **사용자 추가 정보 (파일명 등)**: {file_info or '없음'}\n\n"
        
        f"## 해설 (Caption) 작성 지침\n"
        f"1. **시점 및 주체**: 해설은 반드시 **1인칭 주체('-했어'체)**로 작성해야 합니다. '내', '우리', '나' 등 주체가 명시된 경우 이를 적극 반영하며, 사용자 추가 정보에 주체(예: 내 강아지, 우리 가족)가 있다면 이를 최우선으로 반영합니다.\n"
        f"2. **구체성 및 요약**: 첫 문장은 사용자 추가 정보를 중심으로 사진의 **주체, 장소, 시간대, 인원수**와 같은 주요 내용을 **가장 간결하고 구체적**으로 요약합니다. (예: '강아지' 대신 '내 강아지 초코', '바다' 대신 '강릉의 바다'.)\n"
        f"3. **상세 묘사**: 첫 문장 다음부터는 사진 속 시각적 요소(행동, 색상, 구도 등)를 **상세히 설명**합니다. 이미 언급한 내용은 반복하지 않으며, **사진에 명확히 보이지 않는 정보는 추측하지 않고 제외**합니다.\n"
        
        f"## 키워드 (Keywords) 작성 지침\n"
        f"1. **개수**: 사진의 주요 요소를 나타내는 명사 또는 명사구로 구성하며, **최대 10개**를 목표로 합니다. 다만, **사진 정보가 충분하지 않다면 자의적 판단 하에 10개 미만으로 출력**할 수 있습니다.\n"
        f"2. **품질 관리**: 키워드는 **단일 명사형**으로 추출하며, **중복되거나 무의미한 키워드(예: 사진, 이미지, 일상 등)**는 **절대 포함하지 않습니다**.\n"
        f"3. **형식**: 각 키워드는 쉼표(,)로 구분하며, **마지막에 마침표(.)를 찍지 않습니다.**\n"
        
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
        "You are an assistant who describes photos clearly for visually impaired users. "
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
