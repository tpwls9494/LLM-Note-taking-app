# AI Study Helper - LLM Backend

Vision + LLM 기반 AI 학습 도우미 백엔드

## 프로젝트 구조

```
ai-study-helper/
├── app/
│   ├── main.py              # FastAPI 앱
│   ├── core/
│   │   └── config.py        # 설정
│   ├── models/
│   │   └── schemas.py       # Pydantic 모델
│   ├── api/v1/              # API 엔드포인트
│   │   ├── summarize.py     # 기능1: 노트 요약
│   │   ├── explain.py       # 기능2: 개념 설명
│   │   ├── generate.py      # 기능3: 문제 생성
│   │   └── feedback.py      # 기능4: 오답 해설
│   └── rag/                 # RAG 시스템 (TODO)
├── requirements.txt
└── .env.example
```

## 설치 및 실행

### 1. 환경 설정

```bash
# 의존성 설치
pip install -r requirements.txt

# .env 파일 생성
cp .env.example .env
# .env 파일에서 OPENAI_API_KEY 설정
```

### 2. 서버 실행

```bash
# 개발 모드 (자동 리로드)
python -m app.main

# 또는 uvicorn 직접 실행
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 3. API 문서 확인

서버 실행 후:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API 엔드포인트

### 기능 1: 노트 요약
```
POST /api/v1/summarize/
```

### 기능 2: 개념 설명
```
POST /api/v1/explain/
```

### 기능 3: 문제 생성
```
POST /api/v1/generate/
```

### 기능 4: 오답 해설
```
POST /api/v1/feedback/
```

## TODO

- [ ] GPT-5-nano LLM 클라이언트 구현
- [ ] ChromaDB RAG 시스템 구현
- [ ] 4개 기능 실제 로직 구현
- [ ] Qwen3-8B 파인튜닝 (기능 3)
- [ ] Vision 파트와 통합 테스트

## 기술 스택

- **FastAPI**: 백엔드 프레임워크
- **GPT-5-nano**: 기능 1,2,4
- **Qwen3-8B**: 기능 3 (파인튜닝)
- **ChromaDB**: RAG 벡터 DB
- **OpenAI Embeddings**: text-embedding-3-small
