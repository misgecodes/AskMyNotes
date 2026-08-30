from pydantic import BaseModel, Field

class Answer(BaseModel):
    answer: str
    confidence: float = Field(ge=0, le=1, description="Confidence from 0 to 1")
