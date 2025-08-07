# list_gemini_models.py

from decouple import config
import google.generativeai as genai

# .env 파일에서 API 키를 불러와 설정합니다.
gemini_api_key = config('GEMINI_API_KEY')
genai.configure(api_key=gemini_api_key)

print("사용 가능한 Gemini 모델 목록:")
for m in genai.list_models():
    # 'generateContent' 메서드를 지원하는 모델만 필터링합니다.
    if 'generateContent' in m.supported_generation_methods:
        print(f"- {m.name}")