from fastapi import APIRouter, HTTPException
from app.models.schemas import GenerateRequest, GenerateResponse, Problem
from app.core.llm import get_llm_client
from app.rag.vector_store import get_vector_store
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/", response_model=GenerateResponse)
async def generate_problems(request: GenerateRequest):
    """
    기능 3: 문제 생성
    
    - 현재: GPT-5-nano 임시 사용
    - 나중에: Qwen3-8B 파인튜닝 모델로 교체
    - RAG 사용 (유사 문제 참조)
    """
    try:
        logger.info(f"문제 생성 요청: {request.weak_concept} ({request.difficulty}, {request.quantity}개)")
        
        # RAG에서 유사 문제 검색
        vector_store = get_vector_store()
        rag_results = vector_store.search(
            query=f"{request.weak_concept} {request.difficulty} 문제",
            n_results=3,
            filter_metadata={"type": "problem"}  # 문제만 검색
        )
        
        # 유사 문제 맥락 생성
        similar_problems = ""
        if rag_results["documents"]:
            similar_problems = "\n\n".join([
                f"예시 {i+1}:\n{doc}" 
                for i, doc in enumerate(rag_results["documents"])
            ])
            logger.info(f"RAG 검색 결과: {len(rag_results['documents'])}개 유사 문제")
        
        # LLM으로 문제 생성
        llm = get_llm_client()
        problems = llm.generate_problems(
            weak_concept=request.weak_concept,
            difficulty=request.difficulty,
            quantity=request.quantity,
            similar_problems=similar_problems
        )
        
        # Pydantic 모델로 변환
        problem_objects = [Problem(**p) for p in problems]
        
        return GenerateResponse(problems=problem_objects)
        
    except Exception as e:
        logger.error(f"문제 생성 실패: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
