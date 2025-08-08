# captioning_module/views.py

import os
import openai
from datetime import timezone
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import ImageSerializer
from .models import Image, DailyTokenUsage
from decouple import config
from django.utils import timezone
import google.generativeai as genai
from PIL import Image as PILImage
from io import BytesIO
from .model import image_captioner
import numpy as np


# --- API 키 설정 ---
try:
    gemini_api_key = config("GEMINI_API_KEY")
    genai.configure(api_key=gemini_api_key)
    print("Gemini API configured successfully.")
except Exception as e:
    print(f"Error configuring Gemini API: {e}")
    gemini_api_key = None

try:
    openai.api_key = config("CHATGPT_API_KEY")
    print("ChatGPT API configured successfully.")
except Exception as e:
    print(f"Error configuring ChatGPT API: {e}")
    openai.api_key = None

# --- 일일 토큰 제한 설정 ---
DAILY_TOKEN_LIMIT = 50000


# --- Embedding 생성 함수 ---
def get_embedding(text, model="text-embedding-3-small"):
    try:
        emb_response = openai.embeddings.create(model=model, input=text)
        return np.array(emb_response.data[0].embedding)
    except Exception as e:
        print(f"Embedding 생성 실패: {e}")
        return None


# --- 벡터 기반 가중치 적용 후 결합 ---
def combine_inputs_with_weights(
    original_caption, user_voice, weight_original=0.7, weight_voice=0.3
):
    emb_caption = get_embedding(original_caption)
    emb_voice = get_embedding(user_voice)

    if emb_caption is None or emb_voice is None:
        return f"{original_caption} {user_voice}"  # fallback: 단순 결합

    weighted_vector = (weight_original * emb_caption) + (weight_voice * emb_voice)

    # 벡터를 직접 텍스트로 변환할 수 없으므로, LLM에 의미를 설명해 주는 방식 사용
    combined_text = (
        f"다음은 두 정보를 {int(weight_original*100)}%:{int(weight_voice*100)}% 비율로 반영한 의미입니다.\n"
        f"원본 캡션: {original_caption}\n"
        f"사용자 설명: {user_voice}\n"
        f"(위 의미는 벡터 결합으로 생성됨)"
    )

    return combined_text


# --- LLM 연동 및 토큰 사용량 체크 함수 (Gemini) ---
def get_refined_caption_with_gemini(original_caption, user_voice_text):
    today = timezone.localdate()
    usage, _ = DailyTokenUsage.objects.get_or_create(date=today)

    if usage.input_tokens + usage.output_tokens >= DAILY_TOKEN_LIMIT:
        return "일일 토큰 사용량 제한에 도달했습니다. 내일 다시 시도해주세요."

    prompt = (
        f"당신은 시각 장애인인 사용자의 요청에 따라 이미지 캡션을 더 자연스럽고 상세하게 다듬어주는 AI 봇입니다.\n"
        f"이미지 캡션: '{original_caption}'\n"
        f"사용자의 추가 설명: '{user_voice_text}'\n"
        f"두 정보를 조합하여, 감성적이고 다채로운 자연스러운 한글 캡션을 생성해주세요."
    )

    estimated_input_tokens = len(prompt.split()) * 2
    if (
        usage.input_tokens + usage.output_tokens + estimated_input_tokens
        >= DAILY_TOKEN_LIMIT
    ):
        return "일일 토큰 사용량 제한에 도달했습니다. 내일 다시 시도해주세요."

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        refined_caption = response.text

        usage.input_tokens += estimated_input_tokens
        usage.output_tokens += len(refined_caption.split()) * 2
        usage.save()

        return refined_caption
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return f"LLM API 호출 실패: {e}"


# --- LLM 연동 및 토큰 사용량 체크 함수 (ChatGPT) ---
def get_refined_caption_with_chatgpt(original_caption, user_voice_text):
    today = timezone.localdate()
    usage, _ = DailyTokenUsage.objects.get_or_create(date=today)

    # ChatGPT는 토큰 사용량을 좀 더 정확하게 계산할 수 있지만, 여기서는 간략하게 처리합니다.
    estimated_tokens = 200  # 예시로 고정값 사용

    if usage.input_tokens + usage.output_tokens + estimated_tokens >= DAILY_TOKEN_LIMIT:
        return "일일 토큰 사용량 제한에 도달했습니다. 내일 다시 시도해주세요."

    prompt = (
        f"당신은 시각 장애인인 사용자의 요청에 따라 이미지 캡션을 더 자연스럽고 상세하게 다듬어주는 AI 봇입니다.\n"
        f"이미지 캡션: '{original_caption}'\n"
        f"사용자의 추가 설명: '{user_voice_text}'\n"
        f"두 정보를 조합하여, 감성적이고 다채로운 자연스러운 한글 캡션을 생성해주세요."
    )

    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        refined_caption = response.choices[0].message.content

        # 실제 토큰 사용량 업데이트 (API 응답에서 토큰 정보를 가져올 수 있습니다)
        usage.input_tokens += response.usage.prompt_tokens
        usage.output_tokens += response.usage.completion_tokens
        usage.save()

        return refined_caption
    except Exception as e:
        print(f"Error calling ChatGPT API: {e}")
        return f"ChatGPT API 호출 실패: {e}"


def get_refined_caption_with_chatgpt_weighted(
    original_caption, user_voice_text, w1=0.7, w2=0.3
):
    combined_input = combine_inputs_with_weights(
        original_caption, user_voice_text, w1, w2
    )

    prompt = (
        f"다음 내용을 참고하여 감성적이고 자연스러운 한글 이미지 캡션을 생성하세요:\n"
        f"{combined_input}"
    )

    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant for image captioning.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"ChatGPT API 호출 실패: {e}"


class ImageCaptioningView(APIView):
    def post(self, request, *args, **kwargs):
        image_file = request.FILES.get("image")
        if not image_file:
            return Response(
                {"error": "No image file provided."}, status=status.HTTP_400_BAD_REQUEST
            )

        weight_original = float(request.data.get("weight_original", 0.7))
        weight_voice = float(request.data.get("weight_voice", 0.3))

        # --- 사용할 LLM을 선택합니다. ("gemini" 또는 "chatgpt") ---
        llm_choice = "gemini"
        if llm_choice == "chatgpt_weighted":
            refined_caption = get_refined_caption_with_chatgpt_weighted(
                original_caption_for_llm, user_voice_text, weight_original, weight_voice
            )

        try:
            image_data = image_file.read()
            analysis_result = image_captioner.analyze_image(image_data, {})

            blip_text = analysis_result.get("file_description", "캡션 생성 실패")
            clip_moods = analysis_result.get("file_moods", [])
            clip_text = clip_moods[0]["label"] if clip_moods else "분위기 분석 실패"

        except Exception as e:
            return Response(
                {"error": f"Error analyzing image with new model: {e}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        user_voice_text = request.data.get("user_voice", "사용자 음성 없음")

        original_caption_for_llm = f"이미지 설명: {blip_text}. 분위기: {clip_text}."

        if llm_choice == "gemini":
            refined_caption = get_refined_caption_with_gemini(
                original_caption_for_llm, user_voice_text
            )
        elif llm_choice == "chatgpt":
            if not openai.api_key:
                return Response(
                    {"error": "ChatGPT API key is not configured."},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                )
            refined_caption = get_refined_caption_with_chatgpt(
                original_caption_for_llm, user_voice_text
            )
        else:
            return Response(
                {"error": "Invalid LLM choice."}, status=status.HTTP_400_BAD_REQUEST
            )

        data_to_save = {
            "image_path": image_file.name,
            "refined_caption": refined_caption,
            "blip_text": blip_text,
            "clip_text": clip_text,
            "user_voice_text": user_voice_text,
            "latitude": request.data.get("latitude"),
            "longitude": request.data.get("longitude"),
            "location": request.data.get("location"),
        }

        serializer = ImageSerializer(data=data_to_save)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
