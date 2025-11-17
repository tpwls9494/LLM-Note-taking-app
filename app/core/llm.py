"""
LLM Client for GPT-5-nano
"""
from typing import List, Dict, Any, Optional
from openai import OpenAI
import logging

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class LLMClient:
    """GPT LLM 클라이언트"""
    
    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.GPT_MODEL
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_completion_tokens: int = 2000,
        response_format: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        GPT 생성 요청
        
        Args:
            messages: [{"role": "system/user/assistant", "content": "..."}]
            temperature: 0.0 ~ 1.0 (gpt-5-nano는 무시됨)
            max_completion_tokens: 최대 토큰 수
            response_format: JSON 응답 형식 ({"type": "json_object"})
        
        Returns:
            생성된 텍스트
        """
        try:
            logger.info(f"GPT 요청: {len(messages)}개 메시지, model={self.model}")
            
            kwargs = {
                "model": self.model,
                "messages": messages,
                "max_completion_tokens": max_completion_tokens
            }
            
            # gpt-5-nano는 temperature 지원 안 함
            if "gpt-5-nano" not in self.model:
                kwargs["temperature"] = temperature
            
            if response_format:
                kwargs["response_format"] = response_format
            
            response = self.client.chat.completions.create(**kwargs)
            
            result = response.choices[0].message.content
            logger.info(f"GPT 응답: {len(result)} 글자")
            
            return result
            
        except Exception as e:
            logger.error(f"GPT 요청 실패: {str(e)}", exc_info=True)
            raise
    
    def summarize(
        self,
        subject: str,
        topic: str,
        text: str,
        formulas: List[str],
        highlights: List[str]
    ) -> Dict[str, Any]:
        """
        기능 1: 노트 요약
        Temperature: 0.3 (일관성)
        """
        system_prompt = """당신은 학생의 노트를 요약하는 AI 어시스턴트입니다.
핵심 개념, 주요 내용, 중요 공식을 간결하게 정리해주세요.

응답은 반드시 다음 JSON 형식으로 해주세요:
{
  "핵심_개념": ["개념1", "개념2", "개념3"],
  "주요_내용": "요약된 내용",
  "중요_공식": ["공식1", "공식2"]
}"""
        
        user_prompt = f"""과목: {subject}
주제: {topic}

노트 내용:
{text}

공식:
{', '.join(formulas) if formulas else '없음'}

하이라이트:
{', '.join(highlights) if highlights else '없음'}

위 노트를 요약해주세요."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = self.generate(
            messages=messages,
            temperature=settings.TEMPERATURE_SUMMARY,
            response_format={"type": "json_object"}
        )
        
        import json
        return json.loads(response)
    
    def explain(
        self,
        concept: str,
        student_level: str,
        difficulty_reason: Optional[str],
        related_context: str = ""
    ) -> Dict[str, Any]:
        """
        기능 2: 개념 설명
        Temperature: 0.7 (창의적 비유)
        """
        system_prompt = """당신은 개념을 쉽게 설명하는 교육 전문가입니다.
3단계로 설명해주세요:
1. 일상적인 비유
2. 구체적인 예시
3. 정확한 정의

응답은 반드시 다음 JSON 형식으로 해주세요:
{
  "step1_비유": "일상생활 비유 설명",
  "step2_예시": "구체적인 예시",
  "step3_정의": "정확한 정의",
  "관련_개념": ["관련개념1", "관련개념2"]
}"""
        
        user_prompt = f"""개념: {concept}
학생 수준: {student_level}"""
        
        if difficulty_reason:
            user_prompt += f"\n어려워하는 이유: {difficulty_reason}"
        
        if related_context:
            user_prompt += f"\n\n이전 학습 내용:\n{related_context}"
        
        user_prompt += "\n\n위 개념을 3단계로 설명해주세요."
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = self.generate(
            messages=messages,
            temperature=settings.TEMPERATURE_EXPLAIN,
            response_format={"type": "json_object"}
        )
        
        import json
        return json.loads(response)
    
    def generate_problems(
        self,
        weak_concept: str,
        difficulty: str,
        quantity: int,
        similar_problems: str = ""
    ) -> List[Dict[str, Any]]:
        """
        기능 3: 문제 생성 (임시, 나중에 Qwen3로 교체)
        Temperature: 0.7
        """
        system_prompt = """당신은 수학 문제를 생성하는 전문가입니다.
한국 교육과정에 맞는 문제를 만들어주세요.

각 문제는 다음을 포함해야 합니다:
- 명확한 문제
- 정확한 답
- 상세한 풀이
- 학생이 주의해야 할 함정

응답은 반드시 다음 JSON 형식으로 해주세요:
{
  "problems": [
    {
      "problem": "문제 내용",
      "answer": "답",
      "solution": "풀이 과정",
      "trap": "주의할 점",
      "difficulty": "쉬움/중간/어려움"
    }
  ]
}"""
        
        user_prompt = f"""취약 개념: {weak_concept}
난이도: {difficulty}
문제 개수: {quantity}"""
        
        if similar_problems:
            user_prompt += f"\n\n유사 문제 예시:\n{similar_problems}"
        
        user_prompt += f"\n\n{quantity}개의 문제를 생성해주세요."
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = self.generate(
            messages=messages,
            temperature=0.7,
            max_completion_tokens=3000,
            response_format={"type": "json_object"}
        )
        
        import json
        result = json.loads(response)
        return result.get("problems", [])
    
    def feedback(
        self,
        problem: str,
        correct_answer: str,
        student_answer: str,
        student_work: Optional[str]
    ) -> Dict[str, Any]:
        """
        기능 4: 오답 해설
        Temperature: 0.7 (따뜻한 톤)
        """
        system_prompt = """당신은 10년차 친절한 수학 선생님입니다.
학생의 오답을 따뜻하게 피드백해주세요.

구조:
1. 칭찬 (잘한 점)
2. 실수 지점
3. 왜 틀렸는지
4. 올바른 방법
5. 다음 팁

응답은 반드시 다음 JSON 형식으로 해주세요:
{
  "praise": "칭찬",
  "mistake": "실수 지점",
  "reason": "틀린 이유",
  "correct_method": "올바른 풀이",
  "tip": "다음 팁"
}"""
        
        user_prompt = f"""문제: {problem}

정답: {correct_answer}
학생 답안: {student_answer}"""
        
        if student_work:
            user_prompt += f"\n학생 풀이:\n{student_work}"
        
        user_prompt += "\n\n따뜻한 피드백을 주세요."
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = self.generate(
            messages=messages,
            temperature=settings.TEMPERATURE_FEEDBACK,
            response_format={"type": "json_object"}
        )
        
        import json
        return json.loads(response)


# 싱글톤 인스턴스
_llm_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """LLM 클라이언트 싱글톤"""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client