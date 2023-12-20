import os

from dotenv import load_dotenv, find_dotenv

from src.models.utility_models import InferenceEngineType, ModelType
from src.services.data import load_db_client, load_chroma_client
from src.services.inference.inference_service import SQLAndVectorInferenceService, VectorInferenceService, \
    InferenceService
from src.services.inference.model_loader import LocalGGufModelLoader

# Load Environment Variables
load_dotenv(find_dotenv('.env'))

_defaultConfig = {
    "db_uri": os.getenv("CLIENT_DB_URI"),
    "vector_db_uri": os.getenv("CHROMA_PATH"),
}


class InferenceServiceFactory:
    _model_instance_map = {
        f"{ModelType.HUGGING_FACE_GGUF}": None,
        f"{ModelType.HUGGING_FACE}": None,
    }
    _local_model_path = os.getenv("LOCAL_GGUF_MODEL_PATH")
    _hugging_face_model_path = os.getenv("LOCAL_HF_MODEL_PATH")  # Not supported due to hardware resource limitations

    def create_inference_service(self, model_type: ModelType, rag_engine_type: InferenceEngineType,
                                 config=None) -> InferenceService:
        """
        Creates an instance or returns an existing instance of an inference service based on the RAG engine type
        Model Type is currently not supported due to limited support for high-end computing hardware

        :param config: Optional Configuration for db uris e.g {db_uri: "C/users/db.db", vector_uri: "/C/users/..."}
        :param model_type: Specifies the model type
        :param rag_engine_type: The RAG engine type
        :return: Returns an instance of the inference service
        """
        config_data = {}
        config_data.update(_defaultConfig)
        if config is not None:
            config_data.update(config)
        vector_database = load_chroma_client(db_uri=config_data["vector_db_uri"])

        if not self._model_instance_map[model_type]:
            model_loader = LocalGGufModelLoader(self._local_model_path)
            self._model_instance_map[model_type] = model_loader.get_model()
            # TODO: Implement ModelType Based Loader to support other model formats (Future development)
            #  only GGUF models are fully supported for inferencing due to GPU computational resource requirements

        if rag_engine_type == InferenceEngineType.VECTOR_AND_SQL:
            # Link DB to existing client data infrastructure
            sql_db_engine = load_db_client(db_uri=config_data["db_uri"])
            return SQLAndVectorInferenceService(model=self._model_instance_map[model_type], vector_db=vector_database,
                                                sql_db_engine=sql_db_engine)
        elif rag_engine_type == InferenceEngineType.VECTOR:
            return VectorInferenceService(model=self._model_instance_map[model_type], vector_db=vector_database)
        else:
            # TODO: Implement a default inference service (Out of current project scope)
            raise ValueError("Invalid engine type specified")
