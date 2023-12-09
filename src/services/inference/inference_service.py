import os

import chromadb
from langchain.schema.language_model import BaseLanguageModel
from llama_index import ServiceContext, Response, SQLDatabase
from llama_index.core import BaseQueryEngine
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.indices.struct_store import NLSQLTableQueryEngine
from llama_index.query_engine import SubQuestionQueryEngine, SQLJoinQueryEngine
from llama_index.tools import QueryEngineTool, ToolMetadata
from sqlalchemy import create_engine

from src.models.utility_models import ModelType, InferenceRAGEngineType
from src.services.inference.model_loader import LocalGGufModelLoader
from src.utils.vector_db_utils import vector_collection_data, get_collection_index


class InferenceService:
    metadata:dict = None
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
            db_path = os.getenv('CHROMA_PATH')
            vector_database = chromadb.PersistentClient(path=db_path)
            embed_model = HuggingFaceEmbedding(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                cache_folder=os.getenv("EMBEDDING_MODEL_CACHE")
            )
            self.service_context = ServiceContext.from_defaults(llm=self.model, embed_model=embed_model)

            # Load the vector collections in the vector store
            vector_query_engines = {}

            # List of query engines from different data sources
            query_engine_tools = []

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
                query_engine_tools.append(query_engine_tool)\

            self.query_engine = SubQuestionQueryEngine.from_defaults(
                service_context=self.service_context,
                query_engine_tools=query_engine_tools
            )
        return self.query_engine
    def generate_response(self, prompt: str) -> Response:
        return self.query_engine.query(prompt)

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
            vector_engine = super()._construct_query_engine()

            # Construct SQL based query engine
            sql_db_uri = os.getenv("INFERENCE_DB_URI")
            sql_engine = create_engine(sql_db_uri)

            # define llama_index sql dependencies
            sql_db = SQLDatabase(engine=sql_engine)
            sql_query_engine = NLSQLTableQueryEngine(sql_database=sql_db, service_context=self.service_context)

            sql_tool = QueryEngineTool.from_defaults(
                query_engine=sql_query_engine,
                description=(
                    "Useful for translating a natural language query into a SQL query over the following four tables:"
                    " company_info, job_post, job_description and job_application with relationships between these four"
                    " tables using join queries and id references where necessary"
                ),
            )
            vec_engine_tool = QueryEngineTool.from_defaults(
                query_engine=vector_engine,
                description=(
                    f"Useful for answering semantic questions about job descriptions and job application resumes based on document embeddings"
                ),
            )

            self.query_engine = SQLJoinQueryEngine(
                sql_tool, vec_engine_tool, service_context=self.service_context
            )
        return self.query_engine


class InferenceServiceFactory:
    service_instance_map = {
        f"{ModelType.HUGGING_FACE_GGUF}": None,
        [ModelType.HUGGING_FACE]: None,
    }
    local_model_path=os.getenv("LOCAL_MODEL_PATH")

    def create_inference_service(self, model_type: ModelType, rag_engine_type: InferenceRAGEngineType) -> InferenceService:
        if not self.service_instance_map[model_type]:
            if model_type == ModelType.HUGGING_FACE:
                model_loader = LocalGGufModelLoader(self.local_model_path)
                self.service_instance_map[model_type] = model_loader.model
            elif model_type == ModelType.HUGGING_FACE_GGUF:


        if rag_engine_type
