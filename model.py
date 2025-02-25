'''
This class defines the creation of fine_tuned model
'''
import time
import gc
import torch
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM, LoraConfig
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)
from trl import SFTTrainer, setup_chat_format, SFTConfig
from evaluation import time_execution

class ModelTrainer:
    def __init__(self, model_name: str, dataset_name: str, prefix_file_name: str = 'finetuned'):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.prefix_file_name = prefix_file_name
        self.model = None
        self.tokenizer = None
        self.dataset = None
    
    def load_model_and_tokenizer(self):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.chat_template = None
        self.tokenizer.padding_side = "right"
        self.model, self.tokenizer = setup_chat_format(self.model, self.tokenizer)

    def load_dataset(self):
        self.dataset = load_dataset("json", data_files=self.dataset_name, split="train")

    def configure_trainer(self):
        max_seq_length = 2048
        args = SFTConfig(
            output_dir=f'./{self.prefix_file_name}_{self.model_name}',
            overwrite_output_dir=True,
            num_train_epochs=9,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,
            gradient_checkpointing=True,
            optim="adamw_torch_fused",
            logging_steps=10,
            save_strategy="epoch",
            learning_rate=2e-4,
            bf16=True,
            tf32=True,
            max_grad_norm=0.3,
            warmup_ratio=0.03,
            lr_scheduler_type="constant",
            max_seq_length=max_seq_length,
            packing=True,
            dataset_kwargs={"add_special_tokens": False, "append_concat_token": False}
        )
        peft_config = LoraConfig(
            lora_alpha=128,
            lora_dropout=0.05,
            r=256,
            bias="none",
            target_modules="all-linear",
            task_type="CAUSAL_LM"
        )
        return SFTTrainer(
            model=self.model,
            args=args,
            train_dataset=self.dataset,
            peft_config=peft_config,
            tokenizer=self.tokenizer,
        )

    def train_and_save(self):
        trainer = self.configure_trainer()
        trainer.train()
        trainer.save_model()

    def finetune_model(self):
        start_time = time.time()
        torch.cuda.empty_cache()
        print(f"---------------------------------------------------------------------------Creating {self.model_name}-------------------------------------------------------------")
        
        self.load_model_and_tokenizer()
        self.load_dataset()
        self.train_and_save()

        # Cleanup
        del self.model, self.tokenizer, self.dataset
        torch.cuda.empty_cache()
        gc.collect()

        end_time = time.time()
        time_taken = end_time - start_time
        print(f"--------------------------------------------------{self.model_name} created in {time_taken} seconds-----------------------------------------------------------------")
        torch.cuda.empty_cache()
        gc.collect()
