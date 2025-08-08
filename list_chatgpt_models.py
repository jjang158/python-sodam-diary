# list_models.py

import openai
from decouple import config

# .env 파일에서 API 키 불러오기
try:
    openai.api_key = config("CHATGPT_API_KEY")
    print("ChatGPT API configured successfully.")
except Exception as e:
    print(f"Error configuring ChatGPT API: {e}")
    openai.api_key = None
    exit()

try:
    models = openai.models.list()
    print("사용 가능한 모델 목록:")
    for model in models.data:
        print(f"- {model.id}")
except Exception as e:
    print(f"Error listing models: {e}")