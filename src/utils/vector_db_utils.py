import json
import os
import re
from dataclasses import asdict
from typing import Union, Optional

from llama_index import VectorStoreIndex, SimpleDirectoryReader, Document
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.objects import SimpleObjectNodeMapping
from llama_index.vector_stores import ChromaVectorStore

from src.models.data_models import JobPost, JobApplication, Company, JobDescription


def get_collection_and_vector_store(db, collection_name: str) -> tuple:
    collection = db.get_or_create_collection(collection_name)
    vec_store = ChromaVectorStore(chroma_collection=collection)
    return collection, vec_store


def clean_vector_db(vector_db_index, collection):
    for ref_doc_id in list(map(lambda a: a['ref_doc_id'], collection.get()["metadatas"])):
        vector_db_index.delete_ref_doc(ref_doc_id)


def get_index(vector_store, service_context) -> VectorStoreIndex:
    return VectorStoreIndex.from_vector_store(vector_store, service_context=service_context)


def get_collection_index(vector_database, collection_name: str, service_context) -> VectorStoreIndex:
    coll, store = get_collection_and_vector_store(vector_database, collection_name)
    return get_index(store, service_context)


def show_collection_data(collection):
    print(list(map(lambda a: a['ref_doc_id'], collection.get()["metadatas"])))


def add_document_to_vector_store(store_index: VectorStoreIndex, file_path: str, entity_ref: str, ref_name: str):
    """Function to create a document with required metadata and storing in vector store with embeddings"""
    document = SimpleDirectoryReader(input_files=[file_path]).load_data()
    if len(document) > 0:
        for doc in document:
            doc.metadata['parent_obj_ref'] = entity_ref
            doc.metadata['ref_name'] = ref_name
            store_index.insert(doc)
        store_index.vector_store.persist("/testing")


def add_object_to_vector_store(store_index: VectorStoreIndex, obj: JobPost | Company | JobApplication | JobDescription,
                               entity_ref: str, ref_name: str):
    """Function to create a document with required metadata and storing in vector store with embeddings"""
    json_output = json.dumps(asdict(obj), indent=0, ensure_ascii=True)
    lines = json_output.split("\n")
    useful_lines = [
        line for line in lines if not re.match(r"^[{}\[\],]*$", line)
    ]
    document = Document(text="\n".join(useful_lines))
    document.metadata['parent_obj_ref'] = entity_ref
    document.metadata['ref_name'] = ref_name
    store_index.insert(document)


embedding_model: Union[HuggingFaceEmbedding, None] = None


def get_embedding_model(model_cache_path: Optional[str] = None) -> HuggingFaceEmbedding:
    global embedding_model
    if embedding_model is None:
        embedding_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            tokenizer_name="sentence-transformers/all-MiniLM-L6-v2",
            cache_folder=model_cache_path if model_cache_path is not None else os.getenv("EMBEDDING_MODEL_CACHE")
        )
    return embedding_model


# Get vector indices
vector_collection_data = [
    {
        "name": "job_description",
        "label": "Job Descriptions"
    }, {
        "name": "resume",
        "label": "Job Applicant Resumes"
    },
    {
        "name": "job_post",
        "label": "Job Posts"
    },
    {
        "name": "company_info",
        "label": "Parent Company Information that contains all Job descriptions, Job Posts, and Job Applications from candidates"
    }
]
