
def init():
    prompt_message = """
    This is a helper utility tool to run scripts on some stub operations. The supported operations are listed below:
    1. Quantize Model (Q8_0): Script runs and quantiles a hugging face model to an 8bit format quantized model.
    2. Download Quantized Model
    """



def _download_model():
    rootChoice = input("Press 1 to use defaults or 2 to specify custom options")
