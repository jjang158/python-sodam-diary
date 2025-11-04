# Dockerfile

# 1. 기본 OS 이미지 설정
FROM python:3.10-slim

# 2. 작업 디렉토리 설정
WORKDIR /app

# 3. 시스템 의존성 설치 (필수)
# Linux에서 Pillow와 같은 이미지 라이브러리를 사용하기 위해 필요합니다.
RUN apt-get update && apt-get install -y \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# 4. 파이썬 의존성 설치 (분리)
# requirements.txt 복사 및 기본 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ⭐⭐⭐ PyTorch CPU 전용 설치 (가장 중요) ⭐⭐⭐
# requirements.txt에 torch 버전이 명시되어 있다면, 여기서 강제로 덮어씌웁니다.
# 최신 PyTorch 버전을 설치하되, --index-url을 사용하여 CPU 전용 설치를 강제합니다.
RUN pip install --no-cache-dir torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 \
    --index-url https://download.pytorch.org/whl/cpu
    

# 5. 소스 코드 복사
COPY . .

# 6. 포트 노출
EXPOSE 8000

# 7. 컨테이너 실행 명령어
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]