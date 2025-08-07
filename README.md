# python-sodam-diary
소담일기(소리를 담는 일기) 서버

# 📷 이미지 캡셔닝 백엔드 서버

### 🌟 프로젝트 소개
이 프로젝트는 사용자가 이미지를 업로드하면, BLIP 모델이 이미지 캡션을 생성하고 Gemini LLM이 이를 더 자연스럽게 다듬어주는 Django 기반의 백엔드 서버입니다.

### ✨ 주요 기능
- **이미지 캡셔닝**: BLIP 모델을 활용한 이미지 분석 및 캡션 생성
- **LLM 기반 캡션 개선**: Gemini LLM을 이용해 캡션을 자연스럽게 다듬음
- **데이터 저장**: 사용자의 이미지 및 관련 데이터를 데이터베이스에 저장
- **API**: RESTful API를 통한 클라이언트-서버 통신

### 🚀 시작하기
아래 단계를 따라 로컬 개발 환경을 구축하고 서버를 실행할 수 있습니다.

1.  **환경 설정**
    -   가상 환경 생성: `python3 -m venv venv`
    -   가상 환경 활성화: `source venv/bin/activate`

2.  **종속성 설치**
    `requirements.txt` 파일에 명시된 모든 패키지를 설치합니다.
    ```bash
    pip install -r requirements.txt
    ```

3.  **.env 파일 설정**
    프로젝트 루트 디렉터리에 `.env` 파일을 생성하고, 발급받은 Gemini API 키를 추가합니다.
    ```env
    GEMINI_API_KEY="[YOUR_GEMINI_API_KEY_HERE]"
    ```

4.  **데이터베이스 마이그레이션**
    ```bash
    python manage.py makemigrations
    python manage.py migrate
    ```

5.  **서버 실행**
    ```bash
    ./runserver.sh
    ```

### 🔗 API 엔드포인트
- **URL**: `POST /api/v1/images/caption/`
- **기능**: 이미지와 사용자 입력을 받아 캡션을 생성하고 저장
- **요청 형식**: `multipart/form-data`
  - `image`: 이미지 파일
  - `user_voice`: (선택) 사용자 음성 텍스트
  - `latitude`, `longitude`: (선택) 위치 정보
- **응답**:
  - `201 Created`
  - JSON 형식으로 저장된 데이터 반환