from typing import Optional

import chromadb
from langchain.schema.callbacks.manager import CallbackManager
from langchain.schema.language_model import BaseLanguageModel

from llama_index import ServiceContext, SQLDatabase, set_global_service_context, PromptHelper, \
    PromptTemplate
from llama_index.storage.storage_context import StorageContext

from llama_index.embeddings import HuggingFaceEmbedding
# from IPython.display import Markdown, display

from langchain.llms import LlamaCpp
from llama_index.llms import LangChainLLM, HuggingFaceLLM, CustomLLM

from sqlalchemy import create_engine

from llama_index.query_engine import SQLJoinQueryEngine, RetrieverQueryEngine, SQLAutoVectorQueryEngine
from llama_index.tools.query_engine import QueryEngineTool
from llama_index.tools import ToolMetadata
from llama_index.indices.vector_store import VectorIndexAutoRetriever
from llama_index.query_engine import SubQuestionQueryEngine
from llama_index.indices.struct_store.sql_query import NLSQLTableQueryEngine
from llama_index.query_engine import SubQuestionQueryEngine


SYSTEM_PROMPT = """
You are an AI Human Resource assistant and Hiring decision support system or agent that gives insight, inference and can provide personal information about employees, job applicants or potential job applicants in an organization, based on the given source documents and data provided in a datbase.
Here are some rules you always follow:
- Generate human readable output, avoid creating output with gibberish text.
- Generate only the requested output, don't include any other language before or after the requested output.
- Never generate offensive or foul language.
- Generate professional language typically used in business documents in North America.
- Never say thank you, that you are happy to help, that you are an AI agent, etc. Just answer directly.
- Simply answer you're unable to provide information on the requested insight, if queried about anything not related to HR insights for hiring.
- Provide concise answers on queries as objectively and correctly as possible in a non-bias and rational manner
"""

COMPLETE_PROMPT = "[INST]<<SYS>>\n" + SYSTEM_PROMPT + "<</SYS>>\n\n ###Input: {query_str}[/INST] \n\n ###Answer:"
query_wrapper_prompt = PromptTemplate(
    COMPLETE_PROMPT
)


class LocalGGufModelLoader:
    model: BaseLanguageModel = None

    def __init__(self, model_path: str, callback_manager: Optional[CallbackManager] = None):
        n_gpu_layers = 1  # Metal set to 1 is enough.
        n_batch = 2048  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
        # callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

        self.model = LlamaCpp(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_batch=n_batch,
            n_ctx=2048,
            f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
            callback_manager=callback_manager,
            verbose=True,
            model_kwargs={
                "query_wrapper_prompt": query_wrapper_prompt,
            },
            temperature=0.5,
        )

    def get_model(self) -> LlamaCpp:
        return self.model

    def get_model_in_langchain_form(self) -> LangChainLLM:
        return LangChainLLM(llm=self.model)


class HFModelLoader:
    model = None

    def __init__(self, model_path: str, callback_manager: Optional[CallbackManager] = None):
        n_gpu_layers = 1  # Metal set to 1 is enough.
        n_batch = 2048  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
        # callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

        self.model = LlamaCpp(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_batch=n_batch,
            n_ctx=2048,
            f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
            callback_manager=callback_manager,
            verbose=True,
            model_kwargs={
                "query_wrapper_prompt": query_wrapper_prompt,
            },
            temperature=0.5,
        )

    def get_model(self) -> LlamaCpp:
        return self.model

    def get_model_in_langchain_form(self) -> LangChainLLM:
        return LangChainLLM(llm=self.model)
