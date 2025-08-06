# captioning_module/tests/test_models.py

from django.test import TestCase
from captioning_module.models import Image  # <-- 이 부분을 이렇게 수정합니다.
from datetime import datetime

class ImageModelTest(TestCase):

    def setUp(self):
        """
        테스트를 시작하기 전에 테스트용 데이터를 미리 생성합니다.
        """
        Image.objects.create(
            image_path='/Users/hong/Projects/so_dam_server/dog.jpeg',
            refined_caption='강아지 사진입니다',
            latitude=37.5665,
            longitude=126.9780
        )
        Image.objects.create(
            image_path='/local/path/to/image2.jpg',
            refined_caption='사용자 설명입니다.',
            user_voice_text='사용자가 말한 내용입니다.'
        )

    def test_image_creation(self):
        """
        Image 객체가 성공적으로 생성되는지 테스트합니다.
        """
        image1 = Image.objects.get(image_path='/Users/hong/Projects/so_dam_server/dog.jpeg')
        image2 = Image.objects.get(user_voice_text='사용자가 말한 내용입니다.')

        self.assertEqual(image1.refined_caption, '강아지 사진입니다')
        self.assertEqual(image2.refined_caption, '사용자 설명입니다.')
        self.assertIsNotNone(image1.created_at)
        
    def test_image_fields_can_be_null(self):
        """
        null=True로 설정된 필드들이 비어있을 때도 객체 생성이 가능한지 테스트합니다.
        """
        image_with_minimal_data = Image.objects.create(
            image_path='/local/path/to/image3.jpg',
            refined_caption='최소한의 데이터로 생성된 이미지입니다.'
        )
        self.assertIsNone(image_with_minimal_data.latitude)
        self.assertIsNone(image_with_minimal_data.location)
        self.assertIsNone(image_with_minimal_data.user_voice_text)