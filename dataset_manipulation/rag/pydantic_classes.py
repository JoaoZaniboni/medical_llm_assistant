from typing import Optional, List
from pydantic import BaseModel, Field


class Response(BaseModel):
    search_result: str 
    source: list
    chunks: list

class SearchQuery(BaseModel):
    query: str
    similarity_top_k: Optional[int] = Field(default=1, ge=1, le=10)

class ResponseText(BaseModel):
    response: str

class EvalQuery(BaseModel):
    similarity_top_k: Optional[int] = Field(default=1, ge=1, le=10)

class EvalResponse(BaseModel):
    hit_rate: float
    mrr : float
    error: str

class ResponseEvalResponse(BaseModel):
    mean_correctness_score: float
    mean_relevancy_score : float
    mean_faithfulness_score: float
    mean_context_similarity_score : float