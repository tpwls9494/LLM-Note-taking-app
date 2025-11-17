# AI Study Helper - LLM Backend

Vision + LLM 기반 AI 학습 도우미 백엔드

## 설치 및 실행

### 1. 환경 설정
```bash
# 가상환경 생성 및 활성화
python3 -m venv venv
source venv/bin/activate

# 의존성 설치
pip install -r requirements.txt

# .env 파일 생성
cp .env.example .env
# .env 파일에서 OPENAI_API_KEY 설정
```

### 2. 서버 실행
```bash
python3 -m app.main
```

### 3. API 문서 확인

- Swagger UI: http://localhost:8000/docs
- 헬스체크: http://localhost:8000/health

## API 엔드포인트

- `POST /api/v1/summarize/` - 노트 요약
- `POST /api/v1/explain/` - 개념 설명
- `POST /api/v1/generate/` - 문제 생성
- `POST /api/v1/feedback/` - 오답 해설

## 기술 스택

- FastAPI
- GPT-5-nano (OpenAI)
- Faiss (RAG)
- Qwen3-8B (문제 생성, 파인튜닝 예정)
