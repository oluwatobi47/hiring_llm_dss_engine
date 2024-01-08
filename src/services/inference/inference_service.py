import json
import logging
import os

from chromadb.api import ClientAPI
from langchain.schema.language_model import BaseLanguageModel
from llama_index import ServiceContext, Response, SQLDatabase, VectorStoreIndex, StorageContext, ComposableGraph
from llama_index.core import BaseQueryEngine
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.indices.struct_store import SQLTableRetrieverQueryEngine
from llama_index.objects import SQLTableNodeMapping, SQLTableSchema, ObjectIndex
from llama_index.query_engine import SubQuestionQueryEngine, SQLJoinQueryEngine
from llama_index.response.schema import StreamingResponse
from llama_index.tools import QueryEngineTool, ToolMetadata
from sqlalchemy import Engine

from src.utils import Benchmarker
from src.utils.vector_db_utils import vector_collection_data, get_collection_index, get_collection_and_vector_store, \
    get_index


class InferenceService:
    metadata: dict = None
    model: BaseLanguageModel = None
    service_context: ServiceContext = None
    query_engine = None

    def _construct_query_engine(self) -> BaseQueryEngine:
        pass

    def generate_response(self, prompt: str) -> str:
        pass

    def refresh_datasource(self):
        """Call when there's changes to datasets"""
        self.query_engine = self._construct_query_engine()

    def get_token_count(self, text: str) -> int:
        return self.model.get_num_tokens(text)


class VectorInferenceService(InferenceService):
    def __init__(self, model, vector_db: ClientAPI, use_json_embeddings=False):
        self.use_json_embeddings = use_json_embeddings
        self.vector_database = vector_db
        self.model = model
        self.metadata = {
            'model_specs': model.config_specs,
            'data_sources': ['Chroma Vector Database']
        }
        self.query_engine = self._construct_query_engine()

    def _construct_composable_graph_query_engine(self, collection_data: list):
        """Constructs a Composable graph Query engine from collection data"""

        # Load the vector collections in the vector store
        vector_stores = {}
        vector_indices = {}
        index_summaries = []

        for info in collection_data:
            name = info['name']
            label = info['label']
            coll, store = get_collection_and_vector_store(self.vector_database, name)
            vector_stores[name] = store
            vector_indices[name] = get_index(store, service_context=self.service_context)
            index_summaries.append(f"Provides detailed information about {label}")
        storage_context = StorageContext.from_defaults(vector_stores=vector_stores)

        graph = ComposableGraph(all_indices=vector_indices, root_id="resume", storage_context=storage_context)
        return graph.as_query_engine(streaming=True)

        # Improved RAG after evaluation increasing top_k similarity and similarity threshold
        # return graph.as_query_engine(
        #     streaming=True,
        #     similarity_top_k=3,
        #     vector_store_query_mode=VectorStoreQueryMode.MMR,
        #     mmr_threshold=0.7,
            # vector_store_kwargs={"mmr_threshold": 0.7, "include":['distances']}
            # vector_store_kwargs={"include":['distances']}
        # )

    def _construct_sub_question_query_engine(self, collection_data):
        """Constructs a SubQuestion Query engine from collection data"""
        # List of query engines from different data sources
        query_engine_tools = []

        # Load the vector collections in the vector store
        vector_query_engines = {}

        for info in collection_data:
            name = info['name']
            label = info['label']
            index = get_collection_index(self.vector_database, name, self.service_context)
            vector_query_engines[name] = index.as_query_engine(service_context=self.service_context, streaming=True)
            query_engine_tool = QueryEngineTool(
                query_engine=vector_query_engines[name],
                metadata=ToolMetadata(
                    name=name, description=f"Provides detailed information about {label}"
                ),
            )
            query_engine_tools.append(query_engine_tool)

        vector_engine = SubQuestionQueryEngine.from_defaults(
            service_context=self.service_context,
            query_engine_tools=query_engine_tools,
        )
        return vector_engine

    def _construct_query_engine(self) -> BaseQueryEngine:
        """
        Uses a composable graph for building a query engine instance for the vector documents
        :return: A BaseQueryEngine
        """
        embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            tokenizer_name="sentence-transformers/all-MiniLM-L6-v2",
            cache_folder=os.getenv("EMBEDDING_MODEL_CACHE")
        )
        self.service_context = ServiceContext.from_defaults(llm=self.model, embed_model=embed_model)

        collection_data = vector_collection_data if self.use_json_embeddings is True else vector_collection_data[:2]

        # Use one of the engines below

        # THIS YIELDS IMPROVED MODEL RESPONSES
        engine = self._construct_sub_question_query_engine(collection_data)

        # OR
        # engine = self._construct_composable_graph_query_engine(collection_data)
        return engine

    def generate_response(self, prompt: str) -> str:
        bm = Benchmarker()
        bm.start()
        result = self.query_engine.query(prompt)
        bm.end()
        print(f">>>Inference execution time: {bm.get_execution_time()}s")
        if isinstance(result, StreamingResponse):
            res = result.get_response()
            return res.response
        elif isinstance(result, Response):
            return result.response

    def refresh_datasource(self):
        self.query_engine = self._construct_query_engine()


class SQLAndVectorInferenceService(VectorInferenceService):
    sql_client_engine = None

    def __init__(self, model, vector_db: ClientAPI, sql_db_engine: Engine):
        self.sql_client_engine = sql_db_engine
        super().__init__(model, vector_db=vector_db)
        self.metadata = {
            'model_specs': model.config_specs,
            'data_sources': ['Chroma Vector Database', 'Relational SQL Database']
        }
        self.query_engine = self._construct_query_engine()

    def _build_sql_query_engine(self):
        # Construct SQL based query engine

        sql_db = SQLDatabase(engine=self.sql_client_engine)
        table_names = ["company_info", "job_description", "job_post", "job_application"]
        table_node_mapping = SQLTableNodeMapping(sql_db)
        table_schema_objs = []

        for table_name in table_names:
            table_schema_objs.append(SQLTableSchema(table_name=table_name))

        obj_index = ObjectIndex.from_objects(
            table_schema_objs,
            table_node_mapping,
            VectorStoreIndex,
            service_context=self.service_context
        )

        sql_query_engine = SQLTableRetrieverQueryEngine(
            sql_db,
            obj_index.as_retriever(similarity_top_k=1),
            service_context=self.service_context
        )
        return sql_query_engine

    def _construct_query_engine(self) -> SQLJoinQueryEngine:
        vector_engine = super()._construct_query_engine()
        sql_query_engine = self._build_sql_query_engine()

        sql_tool = QueryEngineTool.from_defaults(
            query_engine=sql_query_engine,
            description=(
                    "For translating a natural language query into a SQL query over the tables in the database"
                    + "using join queries and id references where necessary"
            ),
        )
        vec_engine_tool = QueryEngineTool.from_defaults(
            query_engine=vector_engine,
            description=("For answering semantic questions about job descriptions and job application"
                         + " resumes based on document embeddings"),
        )

        return SQLJoinQueryEngine(
            sql_tool, vec_engine_tool, service_context=self.service_context
        )
