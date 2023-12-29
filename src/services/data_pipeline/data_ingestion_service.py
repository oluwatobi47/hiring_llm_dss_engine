import os

from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from llama_index import ServiceContext, VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.vector_stores.types import VectorStore

# from chromadb.api.models import Collection

from src.models.data_models import JobApplication, JobDescription, JobPost, Company
from src.utils.vector_db_utils import get_collection_and_vector_store, get_index, add_document_to_vector_store, \
    add_object_to_vector_store


class DataIngestionService:
    """ Class with utility functions for loading data model entries to vector embeddings
    particularly for resume and job description data updates (This version will only support new entries)
    """

    _vector_client_api: ClientAPI = None
    _embedding_model: HuggingFaceEmbedding = None
    _collections_map = {}
    _vector_store_map = {}
    _service_context = None

    def __init__(self, chroma_db_client: ClientAPI, embedding_model: HuggingFaceEmbedding):
        self._vector_client_api = chroma_db_client
        self._embedding_model = embedding_model
        self._service_context = ServiceContext.from_defaults(embed_model=self._embedding_model, llm=None)

    def get_client(self):
        return self._vector_client_api

    def refresh_datasource(self):
        self._collections_map = {}
        self._vector_store_map = {}

    def process_job_application_create(self, job_application: JobApplication):
        if not bool(job_application) or not hasattr(job_application, 'resume_link'):
            print(f"Skipping data processing for Job Application: {job_application}")
            return
        store = self.get_store("resume")
        index = get_index(store, self._service_context)
        add_document_to_vector_store(
            index,
            file_path=job_application.resume_link,
            ref_name=f"{job_application.job_post_id}_{job_application.candidate_name}",
            entity_ref=job_application.id)

    def process_job_description_create(self, job_description: JobDescription):
        if not job_description or not hasattr(job_description, 'job_description_file'):
            print(f"Skipping data processing for Job Description: {job_description}")
            return
        store = self.get_store("job_description")
        index = get_index(store, self._service_context)
        add_document_to_vector_store(
            index,
            file_path=job_description.job_description_file,
            ref_name=f"{job_description.job_title}_{job_description.company_id}",
            entity_ref=job_description.id)

    def process_job_post_create(self, job_post: JobPost):
        store = self.get_store("job_post")
        index = get_index(store, self._service_context)
        add_object_to_vector_store(
            index,
            obj=job_post,
            ref_name=f"{job_post.job_description_id}_{job_post.title}",
            entity_ref=job_post.id)

    def process_company_info_create(self, comp: Company):
        store = self.get_store("company_info")
        index = get_index(store, self._service_context)
        add_object_to_vector_store(
            index,
            obj=comp,
            ref_name=f"{comp.name}",
            entity_ref=comp.id)

    def _process_job_application_update(self, job_application: JobApplication):
        pass

    def _process_job_description_update(self, job_description: JobDescription):
        pass

    def _process_job_application_delete(self, job_application_id: str):
        pass

    def _process_job_description_delete(self, job_description_id: str):
        pass

    def get_collection(self, collection_name: str) -> Collection:
        # collection name should either be "resume" or "job_description"
        if collection_name not in ["resume", "job_description", "company_info", "job_post"]:
            raise ValueError("Unsupported collection name specified!")
        if collection_name not in self._collections_map:
            col, store = get_collection_and_vector_store(self._vector_client_api, collection_name)
            self._collections_map[collection_name] = col
            self._vector_store_map[collection_name] = store
        return self._collections_map[collection_name]

    def get_store(self, collection_name: str) -> VectorStore:
        # collection name should either be "resume" or "job_description"
        if collection_name not in ["resume", "job_description", "company_info", "job_post"]:
            raise ValueError("Unsupported collection name specified!")
        if collection_name not in self._collections_map:
            col, store = get_collection_and_vector_store(self._vector_client_api, collection_name)
            self._collections_map[collection_name] = col
            self._vector_store_map[collection_name] = store
        return self._vector_store_map[collection_name]
