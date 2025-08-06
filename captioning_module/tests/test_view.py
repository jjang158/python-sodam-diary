# captioning_module/tests/test_views.py

from rest_framework.test import APITestCase
from rest_framework import status
from django.urls import reverse
from captioning_module.models import Image
from PIL import Image as PILImage
from io import BytesIO
from decimal import Decimal # <-- 이 줄을 추가합니다.

class ImageCaptioningViewTest(APITestCase):
    def setUp(self):
        # 테스트를 위해 가상의 이미지 파일을 생성합니다.
        image = PILImage.new('RGB', (100, 100), 'white')
        image_bytes = BytesIO()
        image.save(image_bytes, format='JPEG')
        image_bytes.seek(0)
        self.image_file = image_bytes
        self.url = reverse('captioning_module:image-captioning') # <-- 이 부분을 이렇게 수정합니다.

    def test_image_captioning_post_success(self):
        """
        유효한 POST 요청을 보냈을 때 201 상태 코드와
        정상적인 응답이 반환되는지 테스트합니다.
        """
        data = {
            'image': self.image_file,
            'latitude': 37.5665,
            'longitude': 126.9780,
        }
        
        # self.client.post를 사용해 API에 POST 요청을 보냅니다.
        response = self.client.post(self.url, data, format='multipart')

        # 1. 응답 상태 코드가 201 Created인지 확인합니다.
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        
        # 2. 응답 데이터에 'refined_caption'이 포함되어 있는지 확인합니다.
        self.assertIn('refined_caption', response.data)
        
        # 3. 데이터베이스에 Image 객체가 성공적으로 생성되었는지 확인합니다.
        self.assertEqual(Image.objects.count(), 1)
        
        # 4. 저장된 객체의 필드 값이 올바른지 확인합니다.
        saved_image = Image.objects.first()
        self.assertEqual(saved_image.latitude, Decimal('37.5665')) # <-- 이 부분을 수정합니다.
        self.assertEqual(saved_image.longitude, Decimal('126.9780')) # <-- 이 부분을 수정합니다.

        
    def test_image_captioning_post_failure_no_image(self):
        """
        이미지 파일 없이 요청을 보냈을 때 400 Bad Request를 반환하는지 테스트합니다.
        """
        data = {
            'latitude': 37.5665
        }
        
        response = self.client.post(self.url, data, format='multipart')
        
        # 1. 응답 상태 코드가 400 Bad Request인지 확인합니다.
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        
        # 2. 데이터베이스에 객체가 생성되지 않았는지 확인합니다.
        self.assertEqual(Image.objects.count(), 0)