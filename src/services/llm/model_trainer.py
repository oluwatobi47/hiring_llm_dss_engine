import os

import torch
from dotenv import load_dotenv, find_dotenv
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer

from src.services.inference.model_loader import SYSTEM_PROMPT

load_dotenv(find_dotenv('.env'))


class ModelTrainer:
    _training_data = None
    _model_name = None
    _pretrained_model = None
    _peft_config = None
    _tokenizer = None

    def __init__(self, model_name, training_data):
        self._training_data = training_data
        self._model_name = model_name
        self._peft_config = self._load_peft_config()
        self._tokenizer = self._load_model_tokenizer()
        self._pretrained_model = self._load_model()

    def _load_model(self):
        # 4 bit Quantization Configuration for LORA
        bits_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_use_double_quant=False,
        )

        # Load tokenizer and model with QLoRA configuration (Helper Reference: https://gist.github.com/younesbelkada/9f7f75c94bdc1981c8ca5cc937d4a4da#comments)
        compute_dtype = getattr(torch, "float16")

        use_nested_quant = False

        # Check GPU compatibility with bfloat16
        if compute_dtype == torch.float16:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                print("=" * 80)
                print("Your GPU supports bfloat16: accelerate training with bf16=True")
                print("=" * 80)

        # Harware resource configuration:
        # Load the entire model on the GPU 0 (Based on available hardware resource)
        device_map = 'auto'  # Use {"": 0} to load only on GPU

        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            self._model_name,
            quantization_config=bits_config,
            device_map=device_map,
        )
        model.config.use_cache = False
        model.config.temperature = 0.6
        model.config.pretraining_tp = 1
        return model

    def _load_model_tokenizer(self):
        # Load model tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self._model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        return tokenizer

    def _load_peft_config(self) -> LoraConfig:
        pass

    def train(self):
        print(f"Training model >> {self.__class__.__name__}")
        # Output directory where the model predictions and checkpoints will be stored
        output_dir = os.getenv("MODEL_TRAINING_OUT_DIR")

        # Number of training epochs
        num_train_epochs = 1

        # Enable fp16/bf16 training (set bf16 to True with an A100)
        fp16 = False
        bf16 = False

        # Batch size per GPU for training
        per_device_train_batch_size = 4

        # Batch size per GPU for evaluation
        per_device_eval_batch_size = 4

        # Number of update steps to accumulate the gradients for
        gradient_accumulation_steps = 1

        # Enable gradient checkpointing
        gradient_checkpointing = True

        # Maximum gradient normal (gradient clipping)
        max_grad_norm = 0.3

        # Initial learning rate (AdamW optimizer)
        learning_rate = 2e-4

        # Weight decay to apply to all layers except bias/LayerNorm weights
        weight_decay = 0.001

        # Optimizer to use
        optim = "paged_adamw_32bit"

        # Learning rate schedule (constant a bit better than cosine)
        lr_scheduler_type = "constant"

        # Number of training steps (overrides num_train_epochs)
        max_steps = -1

        # Ratio of steps for a linear warmup (from 0 to learning rate)
        warmup_ratio = 0.03

        # Group sequences into batches with same length
        # Saves memory and speeds up training considerably
        group_by_length = True

        # Set training parameters
        training_arguments = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            optim=optim,
            save_steps=25,
            logging_steps=1,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            fp16=fp16,
            bf16=bf16,
            max_grad_norm=max_grad_norm,
            max_steps=max_steps,
            warmup_ratio=warmup_ratio,
            group_by_length=group_by_length,
            lr_scheduler_type=lr_scheduler_type,
            report_to=["tensorboard"]
        )

        # Set supervised fine-tuning parameters
        trainer = SFTTrainer(
            model=self._pretrained_model,
            train_dataset=self._training_data['train'],
            peft_config=self._peft_config,
            dataset_text_field="text",
            max_seq_length=None,
            tokenizer=self._tokenizer,
            args=training_arguments,
            packing=False,
        )

        # Train model
        trainer.train()

        # Save trained model
        new_model_path = os.getenv('ADAPTER_MODEL_TRAINING_OUT_DIR')
        trainer.model.save_pretrained(new_model_path)


class LLamaModelTrainer(ModelTrainer):
    _target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"]
    _peft_config = None

    def __init__(self, model_name, training_data):
        super().__init__(model_name, training_data)

    def _load_peft_config(self) -> LoraConfig:
        return LoraConfig(
            lora_alpha=16,  # Alpha parameter for LoRA scaling
            lora_dropout=0.1,  # Dropout probability for LoRA layers
            r=64,  # LoRA attention dimension
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=self._target_modules
        )


class Mistral7BModelTrainer(ModelTrainer):
    _target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"]

    def __init__(self, model, training_data):
        super().__init__(model, training_data)

    def _load_peft_config(self) -> LoraConfig:
        return LoraConfig(
            lora_alpha=16,  # Alpha parameter for LoRA scaling
            lora_dropout=0.05,  # Dropout probability for LoRA layers
            r=16,  # LoRA attention dimension
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=self._target_modules
        )


def _llama_prompt_generator(prompt, response):
    full_prompt = ""
    full_prompt += "<s> [INST]"
    full_prompt += f"<<SYS>>{SYSTEM_PROMPT}<</SYS>>"
    full_prompt += f"\n\n### Input:"
    full_prompt += f"\n{prompt}[/INST]"
    full_prompt += "\n\n### Answer:"
    full_prompt += f"\n{response}"
    full_prompt += "</s>"
    return full_prompt


def _mistral_prompt_generator(prompt, response):
    full_prompt = ""
    full_prompt += "<s>"
    full_prompt += "### Instruction:"
    full_prompt += f"\n{SYSTEM_PROMPT}"
    full_prompt += "\n\n### Input:"
    full_prompt += f"\n{prompt}"
    full_prompt += "\n\n### Response:"
    full_prompt += f"\n{response}"
    full_prompt += "</s>"
    return full_prompt


def prompt_generator(batch_data, model_engine):
    generator_function = _llama_prompt_generator if model_engine == "llama" else (
        _mistral_prompt_generator if model_engine == "mistral" else None)
    if generator_function is None:
        raise ValueError("Engine or Base Model not specified")
    training_prompt = [generator_function(data[0], data[1]) for data in
                       zip(batch_data['prompt'], batch_data['response'])]
    batch_data['text'] = training_prompt
    return batch_data
