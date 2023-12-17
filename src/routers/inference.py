import traceback
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

from src.models.utility_models import InferenceEngineType, ModelType, ApiResponse
from src.services.inference import InferenceServiceFactory, InferenceService

router = APIRouter()


class InferencePrompt(BaseModel):
    prompt: str
    context: Optional[str] = None


# Initialize inference Service factory
inference_service_factory = InferenceServiceFactory()

# Initialize inference Service based on vector document embeddings and SQL data sources
inference_service: InferenceService = inference_service_factory.create_inference_service(
    model_type=ModelType.HUGGING_FACE_GGUF,
    rag_engine_type=InferenceEngineType.VECTOR_AND_SQL
)

# Initialize inference Service based on vector document embeddings only
document_inference_service: InferenceService = inference_service_factory.create_inference_service(
    model_type=ModelType.HUGGING_FACE_GGUF,
    rag_engine_type=InferenceEngineType.VECTOR
)


@router.post("/generate-inference")
def generate_inference(request: InferencePrompt) -> ApiResponse:
    try:
        response = document_inference_service.generate_response(request.prompt)
    except Exception as e:
        print(e)
        traceback.print_exc()
        return ApiResponse(status="error", message=str(e))
    return ApiResponse(status="success", data=response)
