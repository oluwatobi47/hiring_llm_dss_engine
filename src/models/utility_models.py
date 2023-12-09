from dataclasses import dataclass
from enum import Enum
from typing import Optional

from pydantic import BaseModel
from responses import Response


class ModelType(Enum):
    """
    Enum representing supported model types
    """
    HUGGING_FACE = "HF",
    HUGGING_FACE_GGUF = "GGUF",


class InferenceRAGEngineType(Enum):
    """
    Enum representation for the supported Retrieval augmented generation combinations with
    the language model for inference and data insights
    """
    # SQL = "SQL" (WIP)
    VECTOR = "VECTOR",
    VECTOR_AND_SQL = "VECTOR_AND_SQL"


    """
    Enum representing supported model types
    """
    HUGGING_FACE = "HF",
    HUGGING_FACE_GGUF = "GGUF",


class DownloadModelRequest(BaseModel):
    """
    Model representing model download request from hugging face
    Required as part of deployment or initial setup workflow
    """
    type: ModelType
    repo_id: str
    model_file_name: Optional[str]


@dataclass
class ApiResponse:
    status: Optional[str]
    message: Optional[any] = None
    data: Optional[any] = None

