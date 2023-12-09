from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class InferencePrompt(BaseModel):
    prompt: str
    context: Optional[str] = None

@router.post("/generate-inference")
def generate_inference(request: InferencePrompt):
