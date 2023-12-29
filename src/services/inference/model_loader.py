import logging
from typing import Optional

import torch
from langchain.llms import LlamaCpp
from langchain.schema.callbacks.manager import CallbackManager
from langchain.schema.language_model import BaseLanguageModel
from llama_index import PromptTemplate
from llama_index.llms import LangChainLLM, HuggingFaceLLM

from src.utils import Benchmarker

# from IPython.display import Markdown, display


SYSTEM_PROMPT = """
You are an AI Human Resource assistant and Hiring decision support system or agent that gives insight, inference and can
 provide personal information about employees, job applicants or potential job applicants in an organization,
  based on the given source documents and data provided in a datbase.
Here are some rules you always follow:
- Generate human readable output, avoid creating output with gibberish text.
- Generate only the requested output, don't include any other language before or after the requested output.
- Never generate offensive or foul language.
- Generate professional language typically used in business documents in North America.
- Never say thank you, that you are happy to help, that you are an AI agent, etc. Just answer directly.
- Simply answer you're unable to provide information on the requested insight, if queried about anything not related to
 HR insights for hiring.
- Provide concise answers on queries as objectively and correctly as possible in a non-bias and rational manner
"""

COMPLETE_PROMPT = "[INST]<<SYS>>\n" + SYSTEM_PROMPT + "<</SYS>>\n\n ###Input: {query_str}[/INST] \n\n ###Answer:"
query_wrapper_prompt = PromptTemplate(
    COMPLETE_PROMPT
)


class LocalGGufModelLoader:
    _model: BaseLanguageModel = None

    def __init__(self, model_path: str, callback_manager: Optional[CallbackManager] = None):
        n_gpu_layers = 1  # For Metal set to 1 | For CPU set to 0.
        n_batch = 2048  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
        # Disabling callback_manager based on implementation approach
        # callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

        bench_marker = Benchmarker()
        print("Loading LlamaCPP model")
        bench_marker.start()
        self._model = LlamaCpp(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_batch=n_batch,
            max_tokens=2048,
            n_ctx=3072,  # 2560, 2048, 3072
            f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
            callback_manager=callback_manager,
            verbose=True,
            model_kwargs={
                "query_wrapper_prompt": query_wrapper_prompt,
                "device": torch.device("mps"),
                "max_new_tokens": 2048,
            },
            temperature=0.5,
            streaming=True
        )
        bench_marker.end()
        print(f"LocalGGufModelLoader: Model load execution time: {bench_marker.get_execution_time()}ms")

    def get_model(self) -> LlamaCpp:
        return self._model

    def get_model_in_langchain_form(self) -> LangChainLLM:
        return LangChainLLM(llm=self._model)


class HFModelLoader:
    _model = None

    def __init__(self, model_path: str, callback_manager: Optional[CallbackManager] = None):
        n_gpu_layers = 1  # Metal set to 1 is enough.
        n_batch = 2048  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
        # callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

        bench_marker = Benchmarker()
        bench_marker.start()
        self._model = HuggingFaceLLM(
            context_window=2048,
            max_new_tokens=4096,
            generate_kwargs={"temperature": 0.5, "do_sample": False},
            system_prompt=COMPLETE_PROMPT,
            model_name="oluwatobi-alao/llama2-hiring",
            tokenizer_name="oluwatobi-alao/llama2-hiring",
            device_map="auto",
            stopping_ids=[50278, 50279, 50277, 1, 0],
            tokenizer_kwargs={"max_length": 2048},
            # uncomment this if using CUDA to reduce memory usage
            model_kwargs={"torch_dtype": torch.float16}
        )
        print(f"HFModelLoader: Model load execution time: {bench_marker.get_execution_time()}s")

    def get_model(self) -> LlamaCpp:
        return self._model

    def get_model_in_langchain_form(self) -> LangChainLLM:
        return LangChainLLM(llm=self._model)
