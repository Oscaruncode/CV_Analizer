from typing import Dict, List
from pydantic import BaseModel

class ResponseInput(BaseModel):
    AnswerID: int
    QuestionId: int
    AnswerText: str

class SentimentOutput(BaseModel):
    AnswerID: int
    Sentiment: int  # 0 = Negative, 1 = Positive, 2 = Neutral

class AnalysisRequest(BaseModel):
    questions: Dict[int, str]
    responses: List[ResponseInput]
