# captioning_module/models.py

from django.db import models

class Image(models.Model):
    # 클라이언트 로컬에 저장된 이미지의 경로
    image_path = models.CharField(max_length=255)

    # LLM을 통해 가공된 텍스트 (필수)
    refined_caption = models.TextField()

    # 사용자의 음성 입력 텍스트 (선택 사항)
    user_voice_text = models.TextField(blank=True, null=True)

    # 이미지가 생성된 시각 (자동으로 저장)
    created_at = models.DateTimeField(auto_now_add=True)

    # 이미지 촬영 장소 (선택 사항)
    location = models.CharField(max_length=100, blank=True, null=True)

    # 이미지 촬영 위치의 위도 (선택 사항)
    latitude = models.DecimalField(max_digits=9, decimal_places=6, blank=True, null=True)

    # 이미지 촬영 위치의 경도 (선택 사항)
    longitude = models.DecimalField(max_digits=9, decimal_places=6, blank=True, null=True)

    def __str__(self):
        return f"Image by {self.created_at}"