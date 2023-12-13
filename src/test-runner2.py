from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import LlamaCpp


def _simple_llama():
    llm = LlamaCpp(
        model_path="/Users/tobialao/Desktop/Software_Projects/msc_project/hiring_llm_dss_engine/resources/model/llama2-hiring.Q4_K_M.gguf",
        n_gpu_layers=1,
        n_batch=512,
        n_ctx=2048,
        f16_kv=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        verbose=True,
    )

    llm("The first man on the moon was ... Let's think step by step")


if __name__ == "__main__":
    _simple_llama()

