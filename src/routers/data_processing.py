from fastapi import APIRouter

from src.models.utility_models import ApiResponse

router = APIRouter()

@router.post("/clear/{collection_name}")
def clean_collection_data(collection_name: str, ref_id: str = None) -> ApiResponse:
    try:
        response = document_inference_service.generate_response(request.prompt)
    except Exception as e:
        print(e)
        traceback.print_exc()
        return ApiResponse(status="error", message=str(e))
    return ApiResponse(status="success", data=response)
