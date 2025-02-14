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


def create_model(model_name: str,
                 dataset_name: str = 'oracle_data.json',
                 prefix_file_name: str = 'finetuned'):
    '''
    This function creates finetuned model
    '''
    start_time = time.time()
    torch.cuda.empty_cache()
    print(f"---------------------------------------------------------------------------Creating {model_name}-------------------------------------------------------------")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.chat_template = None
    tokenizer.padding_side = "right"
    model, tokenizer = setup_chat_format(model, tokenizer)
    dataset = load_dataset("json",
                           data_files=dataset_name,
                           split="train")
    max_seq_length = 2048
    args = SFTConfig(
        output_dir=f'./{prefix_file_name}_{model_name}',
        overwrite_output_dir=True,
        num_train_epochs=9,
        per_device_train_batch_size=1,  # Reduce batch size # look into this
        gradient_accumulation_steps=16,  # Increase gradient accumulation steps
        gradient_checkpointing=True,
        optim="adamw_torch_fused", # hyperparameter (can switch)
        logging_steps=10, # not sure
        save_strategy="epoch",
        learning_rate=2e-4,
        bf16=True,
        tf32=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        max_seq_length=max_seq_length,
        packing=True,
        dataset_kwargs={
            "add_special_tokens": False,
            "append_concat_token": False
        }
    )
    peft_config = LoraConfig(
        lora_alpha=128,
        lora_dropout=0.05,
        r=256,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM"
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        peft_config=peft_config,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model()
    del model
    del trainer
    torch.cuda.empty_cache()
    gc.collect()
    model = AutoPeftModelForCausalLM.from_pretrained(
        args.output_dir,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    del model
    del merged_model
    del tokenizer
    end_time = time.time()
    time_taken = end_time - start_time
    print(f"--------------------------------------------------{model_name} created in {time_taken} seconds-----------------------------------------------------------------")
    torch.cuda.empty_cache()
    gc.collect()