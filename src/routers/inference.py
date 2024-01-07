import traceback
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

from src.models.utility_models import InferenceEngineType, ModelType, ApiResponse, InferencePrompt
from src.services.inference import InferenceService, inference_service_factory

router = APIRouter()


# PS/NOTE: The Integration into SQL data sources is currently experimental,
# challenges were observed with inconsistent results due to minimal or low capability of handling entity relations
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
def generate_inference(request: InferencePrompt,
                       engine_type: InferenceEngineType = InferenceEngineType.VECTOR) -> ApiResponse:
    try:
        if engine_type == InferenceEngineType.VECTOR:
            response = document_inference_service.generate_response(request.prompt)
        else:
            response = inference_service.generate_response(request.prompt)
    except Exception as e:
        print(e)
        traceback.print_exc()
        return ApiResponse(status="error", message=str(e))
    return ApiResponse(status="success", data=response)


@router.get("/refresh")
def refresh_datasource(engine_type: InferenceEngineType = InferenceEngineType.VECTOR) -> ApiResponse:
    try:
        if engine_type == InferenceEngineType.VECTOR:
            response = document_inference_service.refresh_datasource()
        else:
            response = inference_service.refresh_datasource()
    except Exception as e:
        print(e)
        traceback.print_exc()
        return ApiResponse(status="error", message=str(e))
    return ApiResponse(status="success", data=response)
