import logging
import os

import chromadb
from dotenv import load_dotenv, dotenv_values, find_dotenv
from langchain.schema.language_model import BaseLanguageModel
from llama_index import ServiceContext, Response, SQLDatabase, VectorStoreIndex, StorageContext, ComposableGraph
from llama_index.core import BaseQueryEngine
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.indices.struct_store import NLSQLTableQueryEngine
from llama_index.query_engine import SubQuestionQueryEngine, SQLJoinQueryEngine
from llama_index.tools import QueryEngineTool, ToolMetadata
from sqlalchemy import create_engine

from src.models.utility_models import ModelType, InferenceEngineType
from src.services.data import load_db_client, load_chroma_client
from src.services.inference.model_loader import LocalGGufModelLoader
from src.utils import Benchmarker
from src.utils.vector_db_utils import vector_collection_data, get_collection_index, get_collection_and_vector_store, \
    get_index

# Load Environment Variables
load_dotenv(find_dotenv('.env'))


class InferenceService:
    metadata: dict = None
    model: BaseLanguageModel = None
    service_context: ServiceContext = None
    query_engine = None

    def _construct_query_engine(self) -> BaseQueryEngine:
        pass

    def generate_response(self, prompt: str) -> str:
        pass


class VectorInferenceService(InferenceService):
    def __init__(self, model):
        self.model = model
        self.metadata = {
            'model_specs': model.config_specs,
            'data_sources': ['Chroma Vector Database']
        }
        self._construct_query_engine()

    def _construct_query_engine(self) -> BaseQueryEngine:
        if not self.query_engine:
            vector_database = load_chroma_client()
            embed_model = HuggingFaceEmbedding(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                tokenizer_name="sentence-transformers/all-MiniLM-L6-v2",
                cache_folder=os.getenv("EMBEDDING_MODEL_CACHE")
            )
            self.service_context = ServiceContext.from_defaults(llm=self.model, embed_model=embed_model)

            # Load the vector collections in the vector store
            vector_stores = {}
            vector_indices = {}
            index_summaries = []

            for info in vector_collection_data:
                name = info['name']
                label = info['label']
                coll, store = get_collection_and_vector_store(vector_database, name)
                vector_stores[name] = store
                vector_indices[name] = get_index(store, service_context=self.service_context)
                index_summaries.append(f"Provides detailed information about {label}")
            storage_context = StorageContext.from_defaults(vector_stores=vector_stores)

            graph = ComposableGraph(all_indices=vector_indices, root_id="resume", storage_context=storage_context)
            self.query_engine = graph.as_query_engine()
        return self.query_engine

    def generate_response(self, prompt: str) -> Response:
        bm = Benchmarker()
        result = bm.benchmark_function(self.query_engine.query, prompt)
        logger = logging.getLogger(self.__class__.__name__)
        logger.info(f"Model load execution time: {bm.get_execution_time()}ms")
        print(f">>>Model load execution time: {bm.get_execution_time()}ms")
        return result.response


class SQLAndVectorInferenceService(VectorInferenceService):
    def __init__(self, model):
        super().__init__(model)
        self.model = model
        self.metadata = {
            'model_specs': model.config_specs,
            'data_sources': ['Chroma Vector Database', 'Relational SQL Database']
        }
        self._construct_query_engine()

    def _construct_query_engine(self) -> BaseQueryEngine:
        if not self.query_engine:

            vector_database = load_chroma_client()
            embed_model = HuggingFaceEmbedding(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                tokenizer_name="sentence-transformers/all-MiniLM-L6-v2",
                cache_folder=os.getenv("EMBEDDING_MODEL_CACHE")
            )
            self.service_context = ServiceContext.from_defaults(llm=self.model, embed_model=embed_model)

            # List of query engines from different data sources
            query_engine_tools = []

            # Load the vector collections in the vector store
            vector_query_engines = {}

            for info in vector_collection_data:
                name = info['name']
                label = info['label']
                index = get_collection_index(vector_database, name, self.service_context)
                vector_query_engines[name] = index.as_query_engine(service_context=self.service_context)
                query_engine_tool = QueryEngineTool(
                    query_engine=vector_query_engines[name],
                    metadata=ToolMetadata(
                        name=name, description=f"Provides detailed information about {label}"
                    ),
                )
                query_engine_tools.append(query_engine_tool)

            vector_engine = SubQuestionQueryEngine.from_defaults(
                service_context=self.service_context,
                query_engine_tools=query_engine_tools
            )

            # Construct SQL based query engine
            sql_db_uri = os.getenv("CLIENT_DB_URI")
            sql_engine = load_db_client(sql_db_uri)
            sql_query_engine_indices = []

            # define llama_index sql dependencies
            #TODO: Complete Implementation for SQL Update
            sql_db = SQLDatabase(engine=sql_engine)
            tables = ["company_info", "job_description", "job_post", "job_application"]
            for table in tables:
                query_engine = NLSQLTableQueryEngine(sql_database=sql_db, service_context=self.service_context,
                                                     tables=[table])
                sql_query_engine_indices.append(query_enginegit)

            # sql_tool = ComposableGraph..from_defaults(
            #     query_engine=sql_query_engine,
            #     description=(
            #             "Useful for translating a natural language query into a SQL query over the following four tables:"
            #             + " company_info, job_post, job_description and job_application with relationships between these"
            #             + " four tables using join queries and id references where necessary"
            #     ),
            # )
            # vec_engine_tool = QueryEngineTool.from_defaults(
            #     query_engine=vector_engine,
            #     description=("Useful for answering semantic questions about job descriptions and job application"
            #                  + " resumes based on document embeddings"),
            # )
            #
            # self.query_engine = SQLJoinQueryEngine(
            #     sql_tool, vec_engine_tool, service_context=self.service_context
            # )
        return self.query_engine


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
