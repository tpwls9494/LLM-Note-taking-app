from fastapi import APIRouter, HTTPException
from app.models.schemas import ExplainRequest, ExplainResponse
from app.core.llm import get_llm_client
from app.rag.vector_store import get_vector_store
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/", response_model=ExplainResponse)
async def explain_concept(request: ExplainRequest):
    """
    기능 2: 개념 설명
    
    - GPT-5-nano 사용
    - Temperature: 0.7 (창의적 비유)
    - RAG 사용 (이전 학습 맥락)
    """
    try:
        logger.info(f"개념 설명 요청: {request.concept} (수준: {request.student_level})")
        
        # RAG에서 관련 학습 내용 검색
        vector_store = get_vector_store()
        rag_results = vector_store.search(
            query=f"{request.concept} 관련 이전 학습",
            n_results=3
        )
        
        # 관련 맥락 생성
        related_context = ""
        if rag_results["documents"]:
            related_context = "\n".join([
                f"- {doc}" for doc in rag_results["documents"]
            ])
            logger.info(f"RAG 검색 결과: {len(rag_results['documents'])}개 문서")
        
        # LLM으로 설명 생성
        llm = get_llm_client()
        result = llm.explain(
            concept=request.concept,
            student_level=request.student_level,
            difficulty_reason=request.difficulty_reason,
            related_context=related_context
        )
        
        return ExplainResponse(**result)
        
    except Exception as e:
        logger.error(f"개념 설명 실패: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
