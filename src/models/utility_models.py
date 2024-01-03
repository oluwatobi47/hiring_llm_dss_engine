import typing

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union, Any

from pydantic import BaseModel

# Declare generic type for applicable models
T = typing.TypeVar('T')


class ModelType(str, Enum):
    """
    Enum representing supported model types
    """
    HUGGING_FACE = "HF",
    HUGGING_FACE_GGUF = "GGUF",


class InferenceEngineType(str, Enum):
    """
    Enum representation for the supported Retrieval augmented generation combinations with
    the language model for inference and data insights
    """
    # SQL = "SQL" (WIP)
    VECTOR = "VECTOR",
    VECTOR_AND_SQL = "VECTOR_AND_SQL"


class DownloadModelRequest(BaseModel):
    """
    Model representing model download request from hugging face
    Required as part of deployment or initial setup workflow
    """
    type: ModelType
    repo_id: str
    model_file_name: Optional[str]


class UploadModelRequest(BaseModel):
    """
    Model representing model download request from hugging face
    Required as part of deployment or initial setup workflow
    """
    model_path: str
    repo_id: str
    model_file_name: Optional[str]


# @dataclass
class ApiResponse(BaseModel, typing.Generic[T]):
    status: str = "success"
    message: Optional[Union[str, None]] = None
    data: Optional[Union[T, Any, None]] = None

    def __init__(self, status="success", message: Optional[Union[str, None]] = "",
                 data: Optional[Union[T, Any, None]] = None, *args, **kwargs):
        super().__init__(**kwargs)
        self.status = status
        self.message = message
        self.data = data


class UpdateQA(BaseModel):
    id: int
    value: bool


class InferencePrompt(BaseModel):
    prompt: str
    context: Optional[str] = None

class BatchIds(BaseModel):
    batch_ids: list[int]
