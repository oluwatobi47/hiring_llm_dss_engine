import os

from dotenv import load_dotenv, find_dotenv

from src.models.utility_models import InferenceEngineType, ModelType
from src.services.inference.inference_service import SQLAndVectorInferenceService, VectorInferenceService, \
    InferenceService
from src.services.inference.model_loader import LocalGGufModelLoader

# Load Environment Variables
load_dotenv(find_dotenv('.env'))


class InferenceServiceFactory:
    _service_instance_map = {
        f"{ModelType.HUGGING_FACE_GGUF}": None,
        f"{ModelType.HUGGING_FACE}": None,
    }
    _local_model_path = os.getenv("LOCAL_GGUF_MODEL_PATH")
    _hugging_face_model_path = os.getenv("LOCAL_HF_MODEL_PATH")  # Not supported due to hardware resource limitations

    def create_inference_service(self, model_type: ModelType, rag_engine_type: InferenceEngineType) -> InferenceService:
        """
        Creates an instance or returns an existing instance of an inference service based on the RAG engine type
        Model Type is currently not supported due to limited support for high-end computing hardware

        :param model_type: Specifies the model type
        :param rag_engine_type: The RAG engine type
        :return: Returns an instance of the inference service
        """
        if not self._service_instance_map[model_type]:
            model_loader = LocalGGufModelLoader(self._local_model_path)
            self._service_instance_map[model_type] = model_loader.get_model()
            # TODO: Implement ModelType Based Loader (Future development)

        if rag_engine_type == InferenceEngineType.VECTOR_AND_SQL:
            return SQLAndVectorInferenceService(model=self._service_instance_map[model_type])
        elif rag_engine_type == InferenceEngineType.VECTOR:
            return VectorInferenceService(model=self._service_instance_map[model_type])
        else:
            # TODO: Implement a default inference service (Out of current project scope)
            raise ValueError("Invalid engine type specified")
