import os
from typing import Optional

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

load_dotenv('../.env')


def download_hf_model(
        repo_id: Optional[str] = None,
        file_name: Optional[str] = None,
        local_dir: Optional[str] = None):
    """Download GGUF Model file from hugging face repository uses defaults specified in .env file

    :param repo_id: Reference to the hugging face repository ID
    :param file_name: Name of file in the repository
    :param local_dir: Path on local device where model should be persisted
    """
    repo_id = os.environ.get("GGUF_MODEL_REPO") if repo_id is None else repo_id
    file_name = os.environ.get("GGUF_MODEL_NAME") if file_name is None else file_name
    local_dir = os.environ.get("MODEL_DIR") if local_dir is None else local_dir
    return hf_hub_download(repo_id=repo_id, filename=file_name, local_dir=local_dir, resume_download=True)

