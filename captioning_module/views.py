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
from PIL import Image as PILImage, ImageFile # ImageFile 모듈을 가져옵니다.
from io import BytesIO
import json
from .model import image_captioner

# --- 이미지 파일 처리 설정 ---
# 잘린 이미지 파일도 처리할 수 있도록 설정합니다.
ImageFile.LOAD_TRUNCATED_IMAGES = True
print("Warning: ImageFile.LOAD_TRUNCATED_IMAGES is set to True. Truncated images will be processed with a warning.")

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
        print("토큰 사용량이 일일 제한에 도달했습니다.")
        return "일일 토큰 사용량 제한에 도달했습니다. 내일 다시 시도해주세요."

    # 프롬프트 수정: 친근하고 직관적인 설명을 요청
    prompt = (
        f"당신은 시각 장애인 친구에게 사진을 설명해주는 다정하고 친근한 친구입니다.\n"
        f"이 사진의 원래 캡션은 '{original_caption}'입니다.\n"
        f"친구의 추가 설명은 '{file_info}'입니다.\n"
        f"이 두 정보를 바탕으로, 친구가 눈으로 보는 것처럼 사진의 상황, 분위기, 그리고 느껴지는 감정들을 생생하고 직관적인 언어로 전달해주세요."
    )

    estimated_input_tokens = len(prompt.split()) * 2
    if (
        usage.input_tokens + usage.output_tokens + estimated_input_tokens
        >= DAILY_TOKEN_LIMIT
    ):
        print("예상 토큰 사용량으로 인해 일일 제한에 도달했습니다.")
        return "일일 토큰 사용량 제한에 도달했습니다. 내일 다시 시도해주세요."

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        refined_caption = response.text

        usage.input_tokens += estimated_input_tokens
        usage.output_tokens += len(refined_caption.split()) * 2
        usage.save()

        # 토큰 사용량 출력
        current_total = usage.input_tokens + usage.output_tokens
        remaining_tokens = DAILY_TOKEN_LIMIT - current_total
        print(f"현재 Gemini 토큰 사용량: {current_total} / {DAILY_TOKEN_LIMIT}")
        print(f"남은 토큰: {remaining_tokens}")

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
        print("토큰 사용량이 일일 제한에 도달했습니다.")
        return "일일 토큰 사용량 제한에 도달했습니다. 내일 다시 시도해주세요."

    # 프롬프트 수정: 친근하고 직관적인 설명을 요청
    prompt = (
        f"당신은 시각 장애인 친구에게 사진을 설명해주는 다정하고 친근한 친구입니다.\n"
        f"이 사진의 원래 캡션은 '{original_caption}'입니다.\n"
        f"친구의 추가 설명은 '{file_info}'입니다.\n"
        f"이 두 정보를 바탕으로, 친구가 눈으로 보는 것처럼 사진의 상황, 분위기, 그리고 느껴지는 감정들을 생생하고 직관적인 언어로 전달해주세요."
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
        
        # 토큰 사용량 출력
        current_total = usage.input_tokens + usage.output_tokens
        remaining_tokens = DAILY_TOKEN_LIMIT - current_total
        print(f"현재 ChatGPT 토큰 사용량: {current_total} / {DAILY_TOKEN_LIMIT}")
        print(f"남은 토큰: {remaining_tokens}")

        return refined_caption
    except Exception as e:
        print(f"Error calling ChatGPT API: {e}")
        return f"ChatGPT API 호출 실패: {e}"


class ImageCaptioningView(APIView):
    def post(self, request, *args, **kwargs):
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
            # 파일을 한 번만 읽어 변수에 저장합니다.
            image_data = file.read()
            
            # 이제 파일 객체 대신 `image_data`를 전달합니다.
            analysis_result = image_captioner.analyze_image(image_data)
            
            blip_text = analysis_result.get("file_description", "캡션 생성 실패")
            clip_moods = analysis_result.get("file_moods", [])

            # mood는 3가지 높은 무드가 모두 나올 수 있도록 수정
            if clip_moods:
                # 상위 3개 무드의 라벨만 추출하여 문자열로 조합
                clip_text = ", ".join([mood["label"] for mood in clip_moods])
            else:
                clip_text = "분위기 분석 실패"
            
        except OSError as e:
            # OSError: image file is truncated 에러를 명확하게 로깅하고 사용자에게 알립니다.
            print(f"WARNING: An OSError occurred. The file may be truncated: {e}")
            return Response(
                {
                    "status": status.HTTP_500_INTERNAL_SERVER_ERROR,
                    "message": f"파일 분석 중 오류 발생: 파일이 손상되었을 수 있습니다.",
                    "data": {}
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        except Exception as e:
            # 기타 예외를 처리합니다.
            print(f"Error analyzing file with new model: {e}")
            return Response(
                {
                    "status": status.HTTP_500_INTERNAL_SERVER_ERROR,
                    "message": f"파일 분석 중 오류 발생: {e}",
                    "data": {}
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        file_info = request.data.get("file_info", "사용자 음성 없음")

        # LLM 프롬프트에 모든 분위기 정보를 포함하도록 수정
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
            "file": file.name, # 파일명(문자열)을 사용하도록 수정
            "refined_caption": refined_caption,
            "blip_text": blip_text,
            "clip_text": clip_text, # 수정된 clip_text를 저장
            "file_info": file_info,
            "latitude": request.data.get("latitude"),
            "longitude": request.data.get("longitude"),
            "location": request.data.get("location"),
        }
        print(data_to_save)

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
