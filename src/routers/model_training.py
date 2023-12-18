import os
import traceback
from dataclasses import dataclass

from datasets import load_dataset
from fastapi import APIRouter
from pydantic import BaseModel
from starlette.background import BackgroundTasks

from src.models.utility_models import ApiResponse
from src.services.llm.model_trainer import LLamaModelTrainer, prompt_generator
from src.services.llm.training import generate_fine_tuning_data

router = APIRouter()


@dataclass
class TrainModelRequest(BaseModel):
    dataset_path: str


@router.post("/peft-fine-tune")
def train_model(request: TrainModelRequest, bg_tasks: BackgroundTasks) -> ApiResponse:
    try:
        if not request.dataset_path.endswith(".json"):
            raise ValueError("dataset_path should be a valid path to a json file")
        dataset = load_dataset("json", data_files=request.dataset_path)
        processed_dataset = prompt_generator(dataset, "llama")
        print(type(processed_dataset))

        # TODO: Uncomment when running on powerful computing device (GPU: RAM Intensive operation)
        # Set pre-trained foundation model path: This path could be to a local directory or a hugging face repository
        # pretrained_model_path = os.getenv("TRAINING_BASE_MODEL")
        # model_trainer = LLamaModelTrainer(pretrained_model_path, )
        # model_trainer.train()
    except RuntimeError as e:
        print(e)
        return ApiResponse(status="error", message="An exception occurred!")
    return ApiResponse(status="success", message="Training running in background")


@router.get("/generate-tuning-template")
def generate_tuning_data() -> ApiResponse:
    json_resume_path = "/Users/tobialao/Desktop/Software Projects/msc_project/hiring_llm_dss_engine/resources/data/raw" \
                       "-it_support_proffessional.json"
    output = None
    try:
        output = generate_fine_tuning_data(json_resume_path)
    except Exception as e:
        print(e)
        traceback.print_exc()
        return ApiResponse(message="Failed to generate tuning data", status="error")

    return ApiResponse(data=output, message="Generated the template successfully", status="success")
