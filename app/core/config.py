from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """애플리케이션 설정"""
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True
    
    # OpenAI
    OPENAI_API_KEY: str
    GPT_MODEL: str = "gpt-5-nano"
    
    # Temperature 설정
    TEMPERATURE_SUMMARY: float = 0.3
    TEMPERATURE_EXPLAIN: float = 0.7
    TEMPERATURE_FEEDBACK: float = 0.7
    
    # RAG (Simple Vector Store)
    RAG_STORAGE_PATH: str = "./rag_storage"
    
    # Qwen3
    QWEN3_MODEL_PATH: str = "./models/qwen3-8b"
    QWEN3_USE_LOCAL: bool = True
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """설정 싱글톤"""
    return Settings()
