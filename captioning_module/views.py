# captioning_module/views.py

from datetime import timezone
import os
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
from .model import image_captioner  # <-- 새로 추가된 모듈 임포트

# 기존 BLIP 모델 관련 import와 모델 로딩 코드는 삭제합니다.

# Gemini API 키 설정
try:
    gemini_api_key = config("GEMINI_API_KEY")
    genai.configure(api_key=gemini_api_key)
    print("Gemini API configured successfully.")
except Exception as e:
    print(f"Error configuring Gemini API: {e}")
    gemini_api_key = None

# --- 일일 토큰 제한 설정 ---
DAILY_TOKEN_LIMIT = 50000

# --- LLM 연동 및 토큰 사용량 체크 함수 ---
def get_refined_caption_with_gemini(original_caption, user_voice_text):
    # 오늘 날짜의 토큰 사용량 객체를 가져오거나 생성합니다.
    today = timezone.localdate()
    usage, _ = DailyTokenUsage.objects.get_or_create(date=today)

    if usage.input_tokens + usage.output_tokens >= DAILY_TOKEN_LIMIT:
        return "일일 토큰 사용량 제한에 도달했습니다. 내일 다시 시도해주세요."

    # API 호출 전, 프롬프트의 토큰 양을 미리 계산합니다.
    prompt = (
        f"당신은 시각 장애인인 사용자의 요청에 따라 이미지 캡션을 더 자연스럽고 상세하게 다듬어주는 AI 봇입니다.\n"
        f"이미지 캡션: '{original_caption}'\n"
        f"사용자의 추가 설명: '{user_voice_text}'\n"
        f"두 정보를 조합하여, 감성적이고 자연스러운 한글 캡션을 생성하되, 적지 않은 단어의 추가는 되도록 지양해."
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


# 기존 process_image_with_blip 함수는 이제 필요 없으므로 삭제합니다.


class ImageCaptioningView(APIView):
    def post(self, request, *args, **kwargs):
        image_file = request.FILES.get("image")
        if not image_file:
            return Response(
                {"error": "No image file provided."}, status=status.HTTP_400_BAD_REQUEST
            )

        # 1. 새로운 image_captioner 모듈을 사용하여 이미지 분석
        try:
            # 파일 객체는 여러 번 읽을 수 없으므로, .read()로 한 번 읽어 둡니다.
            image_data = image_file.read()
            # analyze_image 함수가 파일 객체를 받는다면 `image_file`을 전달합니다.
            # 지금은 `image_data`를 전달하는 것으로 가정합니다.
            analysis_result = image_captioner.analyze_image(image_data, {})
            
            blip_text = analysis_result.get("file_description", "캡션 생성 실패")
            
            # file_moods 리스트에서 가장 높은 점수의 라벨을 추출합니다.
            clip_moods = analysis_result.get("file_moods", [])
            clip_text = clip_moods[0]["label"] if clip_moods else "분위기 분석 실패"
            
        except Exception as e:
            return Response(
                {"error": f"Error analyzing image with new model: {e}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        # 2. 사용자 음성 입력과 분석 결과를 바탕으로 LLM 후처리 로직 구현
        user_voice_text = request.data.get("user_voice", "사용자 음성 없음")

        # Gemini 프롬프트에 전달할 원본 캡션 조합
        original_caption_for_gemini = f"이미지 설명: {blip_text}. 분위기: {clip_text}."
        
        refined_caption = get_refined_caption_with_gemini(
            original_caption_for_gemini, user_voice_text
        )

        # 3. 모든 분석 결과와 최종 캡션을 DB에 저장
        data_to_save = {
            "image_path": image_file.name,
            "refined_caption": refined_caption,
            "blip_text": blip_text,  # 새로운 필드 추가
            "clip_text": clip_text,  # 새로운 필드 추가
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
