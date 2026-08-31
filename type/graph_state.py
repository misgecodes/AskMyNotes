from typing import TypedDict


class GraphState(TypedDict):
    question : str
    chunks: list[str]
    answer: str 
    confidence: float
    retries : int
    edges: list[tuple[str, str]]