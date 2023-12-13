import os

from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from llama_index import ServiceContext, VectorStoreIndex
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.vector_stores.types import VectorStore

# from chromadb.api.models import Collection

from src.models.data_models import JobApplication, JobDescription
from src.utils.vector_db_utils import get_collection_and_vector_store, get_index, add_document_to_vector_store


class DataIngestionService:
    """ Class with utility functions for loading data model entries to vector embeddings
    particularly for resume and job description data updates (This version will only support new entries)
    """

    _chroma_client_api: ClientAPI = None
    _embedding_model: HuggingFaceEmbedding = None
    _collections_map = {}
    _vector_store_map = {}

    def __init__(self, chroma_db_client: ClientAPI, embedding_model: HuggingFaceEmbedding):
        self._vector_client_api = chroma_db_client
        self._embedding_model = embedding_model

    def process_job_application_create(self, job_application: JobApplication):
        service_context = ServiceContext.from_defaults(embed_model=self._embedding_model, llm=None)
        store = self.get_store("resume")
        index = get_index(store, service_context)
        doc = add_document_to_vector_store(
            index,
            file_path=job_application['resume_link'],
            ref_name=f"{job_application['job_post_id']}_{application['candidate_name']}",
            entity_ref=job_application['id'])
        collection = self.get_collection("resume")

    def process_job_description_create(self, job_description: JobDescription):
        pass

    def _process_job_application_update(self, job_application: JobApplication):
        pass

    def _process_job_description_update(self, job_description: JobDescription):
        pass

    def _process_job_application_delete(self, job_application_id: str):
        pass

    def _process_job_description_delete(self, job_description_id: str):
        pass

    def add_document_to_vector_store(store_index: VectorStoreIndex, file_path: str, entity_ref: str, ref_name: str):
        """Function to create a document with required metadata and storing in vectore store with embeddings"""
        document = SimpleDirectoryReader(input_files=[file_path]).load_data()
        document[0].metadata['parent_obj_ref'] = entity_ref
        document[0].metadata['ref_name'] = ref_name
        store_index.insert(document[0])

    def get_collection(self, collection_name: str) -> Collection:
        # collection name should either be "resume" or "job_description"
        if collection_name not in ["resume", "job_description"]:
            raise ValueError("Unsupported collection name specified!")
        if collection_name not in self._collections_map:
            col, store = get_collection_and_vector_store(self._vector_client_api, collection_name)
            self._collections_map[collection_name] = col
            self._vector_store_map[collection_name] = store
        return self._collections_map[collection_name]

    def get_store(self, collection_name: str) -> VectorStore:
        # collection name should either be "resume" or "job_description"
        if collection_name not in ["resume", "job_description"]:
            raise ValueError("Unsupported collection name specified!")
        if collection_name not in self._collections_map:
            col, store = get_collection_and_vector_store(self._vector_client_api, collection_name)
            self._collections_map[collection_name] = col
            self._vector_store_map[collection_name] = store
        return self._vector_store_map[collection_name]
