from fastapi import APIRouter, HTTPException
from app.models.schemas import FeedbackRequest, FeedbackResponse
from app.core.llm import get_llm_client
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/", response_model=FeedbackResponse)
async def feedback_on_mistake(request: FeedbackRequest):
    """
    기능 4: 오답 해설
    
    - GPT-5-nano 사용
    - Temperature: 0.7 (따뜻한 톤)
    - RAG 사용 안 함
    - 시스템 프롬프트: "10년차 친절한 선생님"
    """
    try:
        logger.info(f"오답 해설 요청: 문제 길이 {len(request.problem)}")
        
        llm = get_llm_client()
        result = llm.feedback(
            problem=request.problem,
            correct_answer=request.correct_answer,
            student_answer=request.student_answer,
            student_work=request.student_work
        )
        
        return FeedbackResponse(**result)
        
    except Exception as e:
        logger.error(f"오답 해설 실패: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
