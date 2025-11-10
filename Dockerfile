# =========================
# 1. Base Image
# =========================
FROM python:3.10-slim

# =========================
# 2. 기본 의존성 설치
# =========================
WORKDIR /app

# Pillow, OpenCV 등 이미지 처리 라이브러리용 의존성
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# =========================
# 3. pip 및 핵심 빌드 환경 준비
# =========================
RUN pip install --upgrade pip setuptools wheel

# numpy는 OpenVINO 호환 버전(1.26.x)을 선행 설치
RUN pip install numpy==1.26.4

# =========================
# 4. requirements.txt 복사 및 설치
# =========================
COPY requirements.txt .

# openvino, optimum-intel이 포함되어 있음
RUN pip install --no-cache-dir -r requirements.txt

# =========================
# 5. PyTorch CPU 전용 설치
# =========================
RUN pip install --no-cache-dir \
    torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/cpu

# =========================
# 6. 소스 복사
# =========================
COPY . .

# =========================
# 7. 포트 및 실행 명령
# =========================
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
