# captioning_module/views.py

from datetime import timezone
import os
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import ImageSerializer
from .models import Image, DailyTokenUsage  # <-- DailyTokenUsage 모델을 import 합니다.
from decouple import config  # <-- decouple 라이브러리를 import 합니다.
from django.utils import timezone
import google.generativeai as genai  # <-- Gemini 라이브러리 추가


# ML 라이브러리 import
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image as PILImage
from io import BytesIO

# --- 서버 시작 시 모델을 한 번만 로드합니다. ---
try:
    # 로컬 서버 테스트를 위해 Hugging Face의 기본 모델을 사용합니다.
    # 나중에 파인튜닝된 모델 경로로 변경합니다.
    MODEL_URI = "Salesforce/blip-image-captioning-large"
    processor = BlipProcessor.from_pretrained(MODEL_URI)
    model = BlipForConditionalGeneration.from_pretrained(MODEL_URI)
    print("BLIP model loaded successfully.")
except Exception as e:
    print(f"Error loading BLIP model: {e}")
    model = None
    processor = None

# Gemini API 키 설정
try:
    gemini_api_key = config("GEMINI_API_KEY")
    genai.configure(api_key=gemini_api_key)
    print("Gemini API configured successfully.")
except Exception as e:
    print(f"Error configuring Gemini API: {e}")
    gemini_api_key = None

# --- 일일 토큰 제한 설정 ---
DAILY_TOKEN_LIMIT = 50000  # <-- 일일 토큰 제한을 50,000으로 설정합니다.


# --- LLM 연동 및 토큰 사용량 체크 함수 ---
def get_refined_caption_with_gemini(original_caption, user_voice_text):
    # 오늘 날짜의 토큰 사용량 객체를 가져오거나 생성합니다.
    today = timezone.localdate()
    usage, _ = DailyTokenUsage.objects.get_or_create(date=today)

    # 현재 사용량이 일일 제한을 초과했는지 확인합니다.
    if usage.input_tokens + usage.output_tokens >= DAILY_TOKEN_LIMIT:
        return "일일 토큰 사용량 제한에 도달했습니다. 내일 다시 시도해주세요."

    # API 호출 전, 프롬프트의 토큰 양을 미리 계산합니다.
    prompt = (
        f"당신은 시각 장애인인 사용자의 요청에 따라 이미지 캡션을 더 자연스럽고 상세하게 다듬어주는 AI 봇입니다.\n"
        f"이미지 캡션: '{original_caption}'\n"
        f"사용자의 추가 설명: '{user_voice_text}'\n"
        f"두 정보를 조합하여, 감성적이고 다채로운 자연스러운 한글 캡션을 생성해주세요."
    )

    # 지금은 실제 토큰 계산을 생략하고, 추후 구현할 예정입니다.
    # 대략적인 토큰 수를 가정하여 제한에 걸리는지 확인합니다.
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

        # API 호출 성공 시 토큰 사용량 업데이트
        # Gemini API는 토큰 사용량을 직접 반환하지 않으므로, 대략적으로 계산하거나 추후 정교한 로직을 추가합니다.
        usage.input_tokens += estimated_input_tokens
        usage.output_tokens += len(refined_caption.split()) * 2
        usage.save()

        return refined_caption
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return f"LLM API 호출 실패: {e}"


# --- 이미지 캡셔닝을 담당하는 함수 ---
def process_image_with_blip(image_data):
    if not model or not processor:
        return "Model not loaded."

    # 1. 이미지 데이터 처리
    pil_image = PILImage.open(BytesIO(image_data)).convert("RGB")

    # 2. BLIP 모델로 캡션 생성
    inputs = processor(pil_image, return_tensors="pt")
    out = model.generate(**inputs)

    # 3. 캡션 디코딩
    original_caption = processor.decode(out[0], skip_special_tokens=True)
    return original_caption


class ImageCaptioningView(APIView):
    def post(self, request, *args, **kwargs):
        image_file = request.FILES.get("image")
        if not image_file:
            return Response(
                {"error": "No image file provided."}, status=status.HTTP_400_BAD_REQUEST
            )

        # 1. BLIP 모델을 사용하여 이미지에서 원본 캡션 생성
        try:
            original_caption = process_image_with_blip(image_file.read())
            if original_caption == "Model not loaded.":
                return Response(
                    {"error": "Failed to load the BLIP model."},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                )
        except Exception as e:
            return Response(
                {"error": f"Error processing image: {e}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        # 2. 사용자 음성 입력과 원본 캡션을 바탕으로 LLM 후처리 로직 구현
        user_voice_text = request.data.get("user_voice", "사용자 음성 없음")
        refined_caption = get_refined_caption_with_gemini(
            original_caption, user_voice_text
        )

        data_to_save = {
            "image_path": "local_device_path_from_client",
            "refined_caption": refined_caption,
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
