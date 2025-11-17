from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging

from app.core.config import get_settings
from app.models.schemas import HealthCheck
from app.api.v1 import summarize, explain, generate, feedback

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 시작/종료 이벤트"""
    # 시작 시
    logger.info("🚀 AI Study Helper 시작")
    logger.info(f"📝 GPT Model: {settings.GPT_MODEL}")
    logger.info(f"🗄️ RAG Storage Path: {settings.RAG_STORAGE_PATH}")
    
    yield
    
    # 종료 시
    logger.info("🛑 AI Study Helper 종료")


# FastAPI 앱 생성
app = FastAPI(
    title="AI Study Helper API",
    description="Vision + LLM 기반 AI 학습 도우미",
    version="1.0.0",
    lifespan=lifespan
)

# CORS 설정 (Vision 파트와 통신용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인만 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========== 헬스체크 ==========

@app.get("/", response_model=HealthCheck)
async def root():
    """API 루트"""
    return HealthCheck(status="healthy")


@app.get("/health", response_model=HealthCheck)
async def health_check():
    """헬스체크"""
    return HealthCheck(status="healthy")


# ========== API v1 라우터 등록 ==========

app.include_router(
    summarize.router,
    prefix="/api/v1/summarize",
    tags=["1. 노트 요약"]
)

app.include_router(
    explain.router,
    prefix="/api/v1/explain",
    tags=["2. 개념 설명"]
)

app.include_router(
    generate.router,
    prefix="/api/v1/generate",
    tags=["3. 문제 생성"]
)

app.include_router(
    feedback.router,
    prefix="/api/v1/feedback",
    tags=["4. 오답 해설"]
)


# ========== 에러 핸들러 ==========

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """HTTP 예외 핸들러"""
    logger.error(f"HTTP Error: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """일반 예외 핸들러"""
    logger.error(f"Unexpected Error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": str(exc)
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )