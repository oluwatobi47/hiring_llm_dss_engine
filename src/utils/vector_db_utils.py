from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores import ChromaVectorStore


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
    """Function to create a document with required metadata and storing in vectore store with embeddings"""
    document = SimpleDirectoryReader(input_files=[file_path]).load_data()
    document[0].metadata['parent_obj_ref'] = entity_ref
    document[0].metadata['ref_name'] = ref_name
    store_index.insert(document[0])


# Get vector indices
vector_collection_data = [{
    "name": "job_description",
    "label": "Job Descriptions"
}, {
    "name": "resume",
    "label": "Job Applicant Resumes"
}]
