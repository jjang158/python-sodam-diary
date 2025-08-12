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
from PIL import Image as PILImage, ImageFile  # ImageFile 모듈을 가져옵니다.
from io import BytesIO
import json
from .model import image_captioner

# --- 이미지 파일 처리 설정 ---
# 잘린 이미지 파일도 처리할 수 있도록 설정합니다.
ImageFile.LOAD_TRUNCATED_IMAGES = True
print(
    "Warning: ImageFile.LOAD_TRUNCATED_IMAGES is set to True. Truncated images will be processed with a warning."
)

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


# 프롬프트 생성 함수: 문자열을 반환하도록 수정
def set_prompt(original_caption, file_info):
    """
    LLM에 전달할 프롬프트 메시지를 생성하여 반환합니다.
    """
    return (
        f"당신은 시각 장애인의 입장에서 일기를 쓰듯 지난 추억을 마치 비장애인이 사진을 보듯 생생하게 회상할 수 있도록 표현합니다.\n"
        f"이 사진의 원래 캡션은 '{original_caption}'입니다.\n"
        f"친구의 추가 설명은 '{file_info}'입니다.\n"
        f"이 두 정보를 바탕으로, 사진의 상황, 분위기, 그리고 느껴지는 감정들을 생생하고 직관적인 언어로 전달하되, 가장 첫 줄은 한 문장의 요약으로 표현하세요."
    )


# 프롬프트 생성 함수: 문자열을 반환하도록 수정
def set_test_prompt(original_caption, file_info):
    """
    LLM에 전달할 프롬프트 메시지를 생성하여 반환합니다.
    """
    blip_text = original_caption.get("file_description", "캡션 생성 실패")
    clip_moods = original_caption.get("file_moods", [])
    return f"""사진 설명 : {blip_text},
            예측된 분위기 (label: 감정, score: 점수): {clip_moods},
            사용자 입력 사진 정보 (선택적): {file_info}"""


# --- LLM 연동 및 토큰 사용량 체크 함수 (Gemini) ---
def get_refined_caption_with_gemini(original_caption, file_info):
    today = timezone.localdate()
    usage, _ = DailyTokenUsage.objects.get_or_create(date=today)

    if usage.input_tokens + usage.output_tokens >= DAILY_TOKEN_LIMIT:
        print("토큰 사용량이 일일 제한에 도달했습니다.")
        return "일일 토큰 사용량 제한에 도달했습니다. 내일 다시 시도해주세요."

    # 프롬프트 생성 함수 호출
    prompt = set_prompt(original_caption, file_info)

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
    estimated_tokens = 200  # 예시로 고정값 사용

    if usage.input_tokens + usage.output_tokens + estimated_tokens >= DAILY_TOKEN_LIMIT:
        print("토큰 사용량이 일일 제한에 도달했습니다.")
        return "일일 토큰 사용량 제한에 도달했습니다. 내일 다시 시도해주세요."

    # 프롬프트 생성 함수 호출
    prompt = set_prompt(original_caption, file_info)

    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": """당신은 시각장애인에게 사진의 내용을 따뜻하고 감각적으로 설명하는 안내자입니다.
                    아래 항목을 참고하여, 생생하고 정서적인 묘사를 포함한 1~3문장의 설명을 만들어주세요.
                    설명에는 시각적 배경, 인물/동물의 행동, 주변 분위기, 감정 등을 포함해,
                    시각장애인이 장면을 머릿속에 떠올릴 수 있게 구성합니다.

                    사진 설명:
                    {사진에 대한 기본 설명 텍스트}

                    예측된 분위기 (label: 감정, score: 점수):
                    {예: [{'label': '평화로움', 'score': 7.75}, {'label': '설렘', 'score': 6.52}, {'label': '고통', 'score': 4.48}]}

                    사용자 입력 사진 정보 (선택적):
                    {사용자가 추가로 설명한 정보, 예: "이 사진은 가족 여행 중 찍은 사진이에요."}

                    출력 형식 예시:
                    “{감정과 분위기 중심으로 부드럽게 시작} {사진에 등장하는 인물과 행동 묘사}, {자연 환경이나 배경 묘사}. {사진을 보는 사람에게 감정적 여운을 주는 마무리 문장}”

                    예시 출력:
                    “노을이 지는 해변에서 평화로움이 감도는 순간이에요.
                    한 여성이 기린과 나란히 앉아 강아지를 쓰다듬으며, 손에 든 휴대전화를 바라보고 있어요.
                    잔잔한 파도와 따뜻한 햇살 속에서, 설렘과 고요함이 함께 머무는 풍경입니다.""",
                },
                {"role": "user", "content": prompt},
            ],
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
                    "data": {},
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        # --- 사용할 LLM을 선택합니다. ("gemini" 또는 "chatgpt") ---
        llm_choice = "chatgpt"

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
                    "data": {},
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
                    "data": {},
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        file_info = request.data.get("file_info", "사용자 음성 없음")

        # LLM 프롬프트에 모든 분위기 정보를 포함하도록 수정
        original_caption_for_llm = f"이미지 설명: {blip_text}. 분위기: {clip_text}."

        if llm_choice == "gemini":
            refined_caption = get_refined_caption_with_gemini(
                # original_caption_for_llm, file_info
                analysis_result,
                file_info,
            )
        elif llm_choice == "chatgpt":
            if not openai.api_key:
                return Response(
                    {
                        "status": status.HTTP_500_INTERNAL_SERVER_ERROR,
                        "message": "ChatGPT API key is not configured.",
                        "data": {},
                    },
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                )
            refined_caption = get_refined_caption_with_chatgpt(
                original_caption_for_llm, file_info
            )
        else:
            return Response(
                {
                    "status": status.HTTP_400_BAD_REQUEST,
                    "message": "Invalid LLM choice.",
                    "data": {},
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        data_to_save = {
            "file": file.name,  # 파일명(문자열)을 사용하도록 수정
            "refined_caption": refined_caption,
            "blip_text": blip_text,
            "clip_text": clip_text,  # 수정된 clip_text를 저장
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
                    "data": serializer.data,
                },
                status=status.HTTP_201_CREATED,
            )
        else:
            return Response(
                {
                    "status": status.HTTP_400_BAD_REQUEST,
                    "message": "Invalid data",
                    "data": serializer.errors,
                },
                status=status.HTTP_400_BAD_REQUEST,
            )
