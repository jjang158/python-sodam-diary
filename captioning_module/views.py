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


# --- LLM 연동 및 토큰 사용량 체크 함수 (Gemini) ---
def get_refined_caption_with_gemini(original_caption, file_info):
    today = timezone.localdate()
    usage, _ = DailyTokenUsage.objects.get_or_create(date=today)

    if usage.input_tokens + usage.output_tokens >= DAILY_TOKEN_LIMIT:
        return "일일 토큰 사용량 제한에 도달했습니다. 내일 다시 시도해주세요."

    prompt = (
        f"당신은 시각 장애인인 사용자의 요청에 따라 이미지 캡션을 더 자연스럽고 상세하게 다듬어주는 AI 봇입니다.\n"
        f"이미지 캡션: '{original_caption}'\n"
        f"사용자의 추가 설명: '{file_info}'\n"
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
def get_refined_caption_with_chatgpt(original_caption, file_info):
    today = timezone.localdate()
    usage, _ = DailyTokenUsage.objects.get_or_create(date=today)

    # ChatGPT는 토큰 사용량을 좀 더 정확하게 계산할 수 있지만, 여기서는 간략하게 처리합니다.
    estimated_tokens = 200 # 예시로 고정값 사용

    if usage.input_tokens + usage.output_tokens + estimated_tokens >= DAILY_TOKEN_LIMIT:
        return "일일 토큰 사용량 제한에 도달했습니다. 내일 다시 시도해주세요."

    prompt = (
        f"당신은 시각 장애인인 사용자의 요청에 따라 이미지 캡션을 더 자연스럽고 상세하게 다듬어주는 AI 봇입니다.\n"
        f"이미지 캡션: '{original_caption}'\n"
        f"사용자의 추가 설명: '{file_info}'\n"
        f"두 정보를 조합하여, 감성적이고 다채로운 자연스러운 한글 캡션을 생성해주세요."
    )

    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
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


class ImageCaptioningView(APIView):
    def post(self, request, *args, **kwargs):
        # file = request.FILES.get("file")
        # file = request.FILES["file"]
        file = request.FILES.get("file")

        if not file:
            return Response(
                {
                    "status": status.HTTP_400_BAD_REQUEST,
                    "message": "File or file information is missing",
                    "data": {}
                },
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # --- 사용할 LLM을 선택합니다. ("gemini" 또는 "chatgpt") ---
        llm_choice = "gemini" 

        try:
            # image_data = file.read()
            # analysis_result = image_captioner.analyze_image(image_data)
            analysis_result = image_captioner.analyze_image(file)
            
            blip_text = analysis_result.get("file_description", "캡션 생성 실패")
            clip_moods = analysis_result.get("file_moods", [])
            clip_text = clip_moods[0]["label"] if clip_moods else "분위기 분석 실패"
            
        except Exception as e:
            return Response(
                {
                    "status": status.HTTP_500_INTERNAL_SERVER_ERROR,
                    "message": f"Error analyzing file with new model: {e}",
                    "data": {}
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        file_info = request.data.get("file_info", "사용자 음성 없음")

        original_caption_for_llm = f"이미지 설명: {blip_text}. 분위기: {clip_text}."
        
        if llm_choice == "gemini":
            refined_caption = get_refined_caption_with_gemini(
                original_caption_for_llm, file_info
            )
        elif llm_choice == "chatgpt":
            if not openai.api_key:
                return Response(
                    {
                        "status": status.HTTP_500_INTERNAL_SERVER_ERROR,
                        "message": "ChatGPT API key is not configured.",
                        "data": {}
                    },
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
            refined_caption = get_refined_caption_with_chatgpt(
                original_caption_for_llm, file_info
            )
        else:
            return Response(
                {
                    "status": status.HTTP_400_BAD_REQUEST,
                    "message": "Invalid LLM choice.",
                    "data": {}
                },
                status=status.HTTP_400_BAD_REQUEST
            )


        data_to_save = {
            "file": file.name,
            "refined_caption": refined_caption,
            "blip_text": blip_text,
            "clip_text": clip_text,
            "file_info": file_info,
            "latitude": request.data.get("latitude"),
            "longitude": request.data.get("longitude"),
            "location": request.data.get("location"),
        }

        serializer = ImageSerializer(data=data_to_save)
        if serializer.is_valid():
            serializer.save()
            return Response(
                {
                    "status": status.HTTP_201_CREATED,
                    "message": "Captioning successful",
                    "data": serializer.data
                },
                status=status.HTTP_201_CREATED
            )
        else:
            return Response(
                {
                    "status": status.HTTP_400_BAD_REQUEST,
                    "message": "Invalid data",
                    "data": serializer.errors
                },
                status=status.HTTP_400_BAD_REQUEST
            )
