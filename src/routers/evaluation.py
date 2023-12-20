import os
import uuid

from dotenv import load_dotenv, find_dotenv
from fastapi import APIRouter
from starlette.background import BackgroundTasks

from src.models.utility_models import ModelType, InferenceEngineType, ApiResponse
from src.services.data import load_chroma_client
from src.services.data_pipeline import DataIngestionService
from src.services.evaluation import SimulationService, MetricsDataService, EvaluationBatch
from src.services.inference import inference_service_factory
from src.utils.vector_db_utils import get_embedding_model

load_dotenv(find_dotenv('.env'))
simulation_vector_db_path = os.getenv("SIMULATION_CHROMA_DB_PATH")
# os.rmdir(simulation_vector_db_path)
chroma_db_client = load_chroma_client(simulation_vector_db_path)

# Run Simulation based on the vector data source layer
# This is due to inconsistent results from the experimental approach of integrating directly with sql data sources
inference_service = inference_service_factory.create_inference_service(
    model_type=ModelType.HUGGING_FACE_GGUF,
    rag_engine_type=InferenceEngineType.VECTOR,
    config={
        "vector_db_uri": simulation_vector_db_path
    }
)
metrics_service = MetricsDataService()
ingestion_service = DataIngestionService(chroma_db_client, get_embedding_model())

# initialize simulation Service
simulation_service = SimulationService(inference_service, ingestion_service, metrics_service)

router = APIRouter()


@router.get("/run-simulation/{test_case_code}")
def run_simulation(test_case_code: str, bg_tasks: BackgroundTasks) -> ApiResponse:
    simulation_batch = None
    error = None
    try:
        simulation_batch = EvaluationBatch(
            batch_ref=str(uuid.uuid4()),
            summary=f"Running simulation based test for {test_case_code}",
            test_case_code=test_case_code,
        )
        metrics_service.get_db_session().add(simulation_batch)
        metrics_service.get_db_session().commit()
        metrics_service.get_db_session().refresh(simulation_batch)
        bg_tasks.add_task(simulation_service.run_simulation, simulation_batch)
    except Exception as e:
        error = str(e)
        print(e)
    finally:
        if error is not None and simulation_batch.id is not None:
            simulation_batch.error = error
            metrics_service.get_db_session().commit()
    return ApiResponse(status="success",
                       message="Operation running in background. Call evaluation batch api to get status updates")


@router.get("/test-cases")
def get_all_test_cases() -> ApiResponse:
    data = simulation_service.get_test_cases()
    return ApiResponse(status="success", data=data)


@router.get("/batches")
def get_all_evaluation_batches() -> ApiResponse:
    data = None
    try:
        data = metrics_service.get_evaluation_batches()
    except Exception as e:
        return ApiResponse(status="success", message=str(e))
    return ApiResponse(status="success", data=data)


@router.get("/batch-results")
def get_all_evaluation_results_cases(batch_nos: list[str], question_nos: list[str]) -> ApiResponse:
    data = None
    try:
        # TODO: Update service call and update service class signature
        data = metrics_service.get_evaluation_batches()
    except Exception as e:
        return ApiResponse(status="success", message=str(e))
    return ApiResponse(status="success", data=data)


@router.get("/operation-metrics")
def get_all_operation_metrics() -> ApiResponse:
    data = None
    try:
        data = metrics_service.get_op_metrics()
    except Exception as e:
        return ApiResponse(status="success", message=str(e))
    return ApiResponse(status="success", data=data)
