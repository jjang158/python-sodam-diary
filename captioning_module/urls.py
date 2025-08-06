# captioning_module/urls.py

from django.urls import path
from .views import ImageCaptioningView

app_name = 'captioning_module' # <-- 이 줄을 추가합니다.

urlpatterns = [
    path('images/caption/', ImageCaptioningView.as_view(), name='image-captioning'),
]