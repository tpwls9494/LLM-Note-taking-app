from fastapi import APIRouter, HTTPException
from app.models.schemas import SummaryRequest, SummaryResponse
from app.core.llm import get_llm_client
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/", response_model=SummaryResponse)
async def summarize_notes(request: SummaryRequest):
    """
    기능 1: 노트 요약
    
    - GPT-5-nano 사용
    - Temperature: 0.3 (일관성)
    - RAG 사용 안 함
    """
    try:
        logger.info(f"노트 요약 요청: {request.subject} - {request.topic}")
        
        llm = get_llm_client()
        result = llm.summarize(
            subject=request.subject,
            topic=request.topic,
            text=request.text,
            formulas=request.formulas or [],
            highlights=request.highlights or []
        )
        
        return SummaryResponse(**result)
        
    except Exception as e:
        logger.error(f"노트 요약 실패: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
