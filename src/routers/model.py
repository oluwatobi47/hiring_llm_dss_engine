from typing import Optional

import fastapi
from fastapi import APIRouter, BackgroundTasks

from src.models.utility_models import DownloadModelRequest, ModelType, ApiResponse
from src.scripts import download_hf_model

router = APIRouter()


@router.post("/download/")
async def download_model(bg_tasks: BackgroundTasks, request: Optional[DownloadModelRequest] = None) -> ApiResponse:
    # output = None
    try:
        if not request:
            bg_tasks.add_task(download_hf_model)  # Download based on defaults in .env file
        elif request.type == ModelType.HUGGING_FACE:
            bg_tasks.add_task(download_hf_model, request.repo_id)
        elif request.type == ModelType.HUGGING_FACE_GGUF:
            bg_tasks.add_task(download_hf_model, request.repo_id, request.model_file_name)
        else:
            raise fastapi.HTTPException(status_code=400, detail="Invalid Model type specified")
    except:
        return ApiResponse(status="error", message="An error occurred!!")

    message = """Download in progress to target directory! 
    Please check target directory when download is completed for model file
    Progress status will be in server logs"""
    return ApiResponse(status="success", message=message)
