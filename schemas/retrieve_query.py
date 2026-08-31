
from pydantic import BaseModel


class ReformulatedQuery(BaseModel):
    new_query: str