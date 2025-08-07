# captioning_module/views.py

import os
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import ImageSerializer

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

# --- 이미지 캡셔닝을 담당하는 함수 ---
def process_image_with_blip(image_data):
    if not model or not processor:
        return "Model not loaded."
    
    # 1. 이미지 데이터 처리
    pil_image = PILImage.open(BytesIO(image_data)).convert('RGB')
    
    # 2. BLIP 모델로 캡션 생성
    inputs = processor(pil_image, return_tensors="pt")
    out = model.generate(**inputs)
    
    # 3. 캡션 디코딩
    original_caption = processor.decode(out[0], skip_special_tokens=True)
    return original_caption

class ImageCaptioningView(APIView):
    def post(self, request, *args, **kwargs):
        image_file = request.FILES.get('image')
        if not image_file:
            return Response({"error": "No image file provided."}, status=status.HTTP_400_BAD_REQUEST)
        
        # 1. BLIP 모델을 사용하여 이미지에서 원본 캡션 생성
        try:
            original_caption = process_image_with_blip(image_file.read())
            if original_caption == "Model not loaded.":
                return Response({"error": "Failed to load the BLIP model."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        except Exception as e:
            return Response({"error": f"Error processing image: {e}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # 2. 사용자 음성 입력과 원본 캡션을 바탕으로 LLM 후처리 로직 구현 (추후)
        # 현재는 원본 캡션을 최종 캡션으로 사용합니다.
        refined_caption = original_caption
        # user_voice_text = request.data.get('user_voice', '') 
        user_voice_text = "이것은 사용자의 음성 입력 더미 데이터입니다."    # 음성 데이터 더미 코드 (추후 삭제)

        data_to_save = {
            'image_path': 'local_device_path_from_client',
            'refined_caption': refined_caption,
            'user_voice_text': user_voice_text,
            'latitude': request.data.get('latitude'),
            'longitude': request.data.get('longitude'),
            'location': request.data.get('location')
        }
        
        serializer = ImageSerializer(data=data_to_save)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)