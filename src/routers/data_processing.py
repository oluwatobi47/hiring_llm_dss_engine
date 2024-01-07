import os
import traceback
from enum import Enum
from typing import Union

from dotenv import load_dotenv, find_dotenv
from fastapi import APIRouter

from src.models.data_models import JobDescription, JobApplication, JobPost, Company
from src.models.utility_models import ApiResponse
from src.services.data import load_chroma_client
from src.services.data_pipeline.data_ingestion_service import DataIngestionService
from src.utils.vector_db_utils import get_embedding_model
from src.routers import inference


class PipelineModel(str, Enum):
    JOB_DESCRIPTION = "JD",
    JOB_APPLICATION = "JA",
    JOB_POST = "JP",
    COMPANY_INFO = "COMP",


router = APIRouter()
chroma_db_client = load_chroma_client()
data_ingestion_service = DataIngestionService(chroma_db_client, get_embedding_model())

load_dotenv(find_dotenv('.env'))
simulation_vector_db_path = os.getenv("SIMULATION_CHROMA_DB_PATH")

simulation_db_client = load_chroma_client(db_uri=simulation_vector_db_path)
simulation_ingestion_service = DataIngestionService(simulation_db_client, get_embedding_model())


@router.post("/clear/{collection_name}")
def clean_collection_data(collection_name: str, ref_id: str = None) -> ApiResponse:
    try:
        client = data_ingestion_service.get_client()
        if ref_id is not None:
            collection = data_ingestion_service.get_collection(collection_name)
            collection.delete(where={
                'ref_doc_id': {
                    '$eq': ref_id
                }
            })
        else:
            client.delete_collection(collection_name)
        data_ingestion_service.refresh_datasource()
    except Exception as e:
        traceback.print_exc()
        return ApiResponse(status="error", message=str(e))
    return ApiResponse(status="success", message=f"Collection {collection_name} data cleared successfully")


@router.post("/reset/vector-simulation-db")
def reset_simulation_vector_data() -> ApiResponse:
    try:
        client = simulation_ingestion_service.get_client()
        client.reset()
    except Exception as e:
        traceback.print_exc()
        return ApiResponse(status="error", message=str(e))
    return ApiResponse(status="success",
                       message=f"Vector Database reset successfully")

@router.post("/reset/vector-db")
def reset_vector_db_data() -> ApiResponse:
    try:
        client = data_ingestion_service.get_client()
        client.reset()
        data_ingestion_service.refresh_datasource()
        inference.inference_service.refresh_datasource()

    except Exception as e:
        traceback.print_exc()
        return ApiResponse(status="error", message=str(e))
    return ApiResponse(status="success",
                       message=f"Vector Database reset successfully")


@router.post("/clear/simulation-db/{collection_name}")
def clean_simulation_collection_data(collection_name: str, ref_id: str = None) -> ApiResponse:
    try:
        client = simulation_ingestion_service.get_client()
        if ref_id is not None:
            collection = simulation_ingestion_service.get_collection(collection_name)
            collection.delete(where={
                'ref_doc_id': {
                    '$eq': ref_id
                }
            })
        else:
            client.delete_collection(collection_name)
            simulation_ingestion_service.refresh_datasource()
    except Exception as e:
        traceback.print_exc()
        return ApiResponse(status="error", message=str(e))
    return ApiResponse(status="success", message=f"Collection {collection_name} data cleared successfully")


@router.post("/process-data/new/{entity_type}")
def process_job_description(entity_type: PipelineModel, request: Union[JobDescription, JobApplication, JobPost, Company]) -> ApiResponse:
    try:
        if entity_type == PipelineModel.JOB_DESCRIPTION:
            data_ingestion_service.process_job_description_create(request)
        elif entity_type == PipelineModel.JOB_APPLICATION:
            data_ingestion_service.process_job_application_create(request)
        elif entity_type == PipelineModel.JOB_POST:
            data_ingestion_service.process_job_post_create(request)
        elif entity_type == PipelineModel.COMPANY_INFO:
            data_ingestion_service.process_company_info_create(request)
    except Exception as e:
        traceback.print_exc()
        return ApiResponse(status="error", message=str(e))
    return ApiResponse(status="success", message=f"Data processed successfully")
