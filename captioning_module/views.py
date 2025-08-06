# captioning_module/views.py

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import ImageSerializer
from .models import Image

# 이 부분에 BLIP, LLM, NLP 모델 연동 로직이 들어갈 예정입니다.
def process_image_and_text(image_data, user_voice):
    # 1. BLIP 모델을 사용하여 image_data에서 캡션 생성
    #    original_caption = blip_model.generate_caption(image_data)
    #
    # 2. original_caption과 user_voice를 바탕으로 LLM에 요청
    #    refined_caption = llm_model.refine_text(original_caption, user_voice)

    # 지금은 더미 데이터 반환
    return "아름다운 풍경 사진입니다.", "사용자 설명입니다."

class ImageCaptioningView(APIView):
    def post(self, request, *args, **kwargs):
        # 클라이언트로부터 multipart/form-data를 통해 이미지 파일을 받습니다.
        image_file = request.FILES.get('image')
        if not image_file:
            return Response({"error": "No image file provided."}, status=status.HTTP_400_BAD_REQUEST)

        # 클라이언트로부터 위도/경도 데이터를 받습니다.
        latitude = request.data.get('latitude')
        longitude = request.data.get('longitude')

        # 이미지와 사용자 음성 입력(향후 추가될 기능)을 처리하는 로직
        # 현재는 더미 데이터로 대체합니다.
        refined_caption, user_voice_text = process_image_and_text(image_file, request.data.get('user_voice'))

        # 데이터베이스에 저장할 데이터를 준비합니다.
        data_to_save = {
            'image_path': 'local_device_path_from_client',  # 클라이언트가 보낸 로컬 경로
            'refined_caption': refined_caption,
            'user_voice_text': user_voice_text,
            'latitude': latitude,
            'longitude': longitude,
            'location': '장소 정보' # NLP 모델로 추출할 예정
        }

        # 시리얼라이저를 사용하여 데이터 유효성 검사 및 저장
        serializer = ImageSerializer(data=data_to_save)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)