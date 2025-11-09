# 프로젝트 소개 (Project Overview)

이 프로젝트는 2단계 API 구조를 통해 사용자가 업로드한 이미지를 분석하고, 사용자 입력과 결합하여 인공지능 기반의 일기 해설과 핵심 키워드를 생성합니다.

## 주요 변경 사항 (Recent Changes)

* **2단계 API 구조 도입:** `/analyze/`와 `/generate/`로 분리하여 효율성과 안정성을 높였습니다.
* **LLM 기반 번역:** Step 1 (`/analyze/`)에서 BLIP 캡션을 LLM을 사용하여 즉시 한국어로 번역하여 반환합니다.
* **DB 의존성 분리:** 현재 DB 저장 로직은 주석 처리되었으며, LLM 결과 반환에 집중합니다.

---

# 개발 환경 설정 (Development Setup)

## 1. 전제 조건 (Prerequisites)

* Docker 및 Docker Compose
* Python 3.10+
* Git
* 유효한 **OpenAI API Key** (또는 Gemini API Key)

## 2. 환경 변수 설정

프로젝트 루트 디렉토리에 `.env` 파일을 생성하고 다음 키를 설정해야 합니다.

```env
# .env 파일 예시
# --- LLM API Keys ---
CHATGPT_API_KEY="YOUR_OPENAI_API_KEY_HERE"
# GEMINI_API_KEY="YOUR_GEMINI_API_KEY_HERE"

# --- Database Setting (현재는 주석 처리되어 사용하지 않음) ---
DATABASE_URL="sqlite+aiosqlite:///./app/sqlite.db"

# --- Token Limit (예시) ---
DAILY_TOKEN_LIMIT=1000000 
````

## 3\. Docker를 이용한 서버 실행

모든 의존성 설치 및 서버 실행은 Docker Compose를 통해 처리됩니다.

```bash
# 최신 코드를 반영하여 빌드하고 백그라운드에서 실행
docker compose up --build -d
```

서버는 \*\*`http://localhost:8000`\*\*에서 실행됩니다.

-----

# API 사용 가이드 (API Usage)

FastAPI의 Swagger UI를 통해 테스트할 수 있습니다. \*\*`http://localhost:8000/docs`\*\*로 접속하세요.

## Step 1: 이미지 분석 (BLIP 캡션 생성)

업로드된 이미지를 분석하고 한국어 캡션을 반환합니다.

  * **엔드포인트:** `POST /api/v1/analyze/`
  * **요청:** `image_file` (Form Data)
  * **응답:**
    ```json
    {
      "caption": "정장 차림의 남성이 스툴에 앉아 있는 모습 (한국어)"
    }
    ```

## Step 2: LLM 해설 및 태그 생성

Step 1의 결과 (`blip_caption`)와 사용자 입력을 결합하여 최종 일기 해설과 태그를 생성합니다.

  * **엔드포인트:** `POST /api/v1/generate/`
  * **요청:** `GenerateRequest` (JSON Body)
  * **응답:**
    ```json
    {
      "diary": "화이트 셔츠와 블랙 팬츠를 입은 남성이 스툴에 앉아 방 안을 바라보고 있습니다. 정장 차림의 남성은 집중한 표정으로 주변을 살펴보고 있습니다.",
      "tags": ["남성", "정장", "스툴", "차분함", "스튜디오", "일산", "사진", "분위기", "앉음", "바라봄"]
    }
    ```

### 요청 Body 예시

```json
{
  "user_input": "오늘 일산 스튜디오에서 찍은 사진",
  "blip_caption": "방 안에 흰 셔츠와 검은 바지를 입은 정장 차림의 남자가 스툴에 앉아 있다",
  "latitude": 37.5665,
  "longitude": 126.9780,
  "location": "서울시청 앞 광장"
}
```

-----

# AWS 배포 환경 (AWS Deployment)

배포 환경에서는 `docker-compose.yml` 파일의 포트 설정을 로컬 환경과 다르게 가져가야 합니다.

  * **로컬 개발:** `ports: "8000:8000"`
  * **EC2 배포:** `ports: "80:8000"` (HTTP 기본 포트 사용)

EC2 배포 시, **`git pull` 전에 로컬 설정을 Stash**하거나 **배포 전용 YAML 파일**을 사용하여 포트 설정을 변경해야 합니다.
