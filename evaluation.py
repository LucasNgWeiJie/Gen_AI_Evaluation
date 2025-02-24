import gc
import json
import time
import torch
import evaluate

from datasets import Dataset
from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer

def time_execution(func):
    """
    Decorator to time the execution of a function and return the time taken.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Start time before the function execution
        result = func(*args, **kwargs)
        end_time = time.time()  # End time after the function execution
        execution_time = end_time - start_time
        print(f"{func.__name__} took {execution_time} seconds")
        return result, execution_time  # Return the result of the function along with the time taken
    return wrapper

class ModelEvaluator:
    def __init__(self, model_name: str, is_finetuned: bool, prefix_file_name: str = 'finetuned'):
        self.model_name = model_name
        self.is_finetuned = is_finetuned
        self.prefix_file_name = prefix_file_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.tokenizer = self.load_model()
    
    @time_execution
    def load_model(self):
        print(f"Loading {self.model_name}.........")
        model_path = f'./{self.prefix_file_name}_{self.model_name}' if self.is_finetuned else self.model_name
        model_cls = AutoPeftModelForCausalLM if self.is_finetuned else AutoModelForCausalLM
        model = model_cls.from_pretrained(model_path).to(self.device)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer

    def clear_memory(self):
        del self.model, self.tokenizer
        torch.cuda.empty_cache()
        gc.collect()

class LossPerplexityEvaluator(ModelEvaluator):
    avg_loss = None
    perplexity = None
    @time_execution
    def evaluate(self, dataset: Dataset):
        print(f"Evaluating Loss perplexity")
        self.model.eval()
        total_loss, total_tokens = 0.0, 0

        for example in dataset:
            inputs = self.tokenizer(example["prompt"], 
                                    return_tensors="pt", 
                                    truncation=True, 
                                    padding=True, 
                                    max_length=512).to(self.device)
            
            targets = self.tokenizer(example["completion"], return_tensors="pt", truncation=True, padding=True, max_length=512).input_ids.to(self.device)
            labels = torch.cat([inputs.input_ids, targets], dim=-1).to(self.device).long()
            
            with torch.no_grad():
                outputs = self.model(input_ids=labels, labels=labels)
                total_loss += outputs.loss.item() * labels.size(1)
                total_tokens += labels.size(1)

        self.avg_loss = total_loss / total_tokens
        self.perplexity = torch.exp(torch.tensor(self.avg_loss))
        self.clear_memory()

class CompletionEvaluator(ModelEvaluator):
    time_completion = None
    time_prompt_token = None
    time_completion_token = None

    def evaluate(self, dataset: Dataset):
        start_time = time.time()
        total_generated, total_prompt_tokens = self.create_completion_dataset(dataset)
        total_time = time.time() - start_time
        num_rows = max(dataset.num_rows, 1)  # Avoid division by zero  
        self.time_completion = total_time / num_rows,
        self.time_prompt_token = total_time / total_prompt_tokens if total_prompt_tokens > 0 else 0,
        self.time_completion_token = total_time / total_generated if total_generated > 0 else 0
        

    def create_completion_dataset(self, dataset: Dataset):
        total_generated, total_prompt_tokens = 0, 0
        for example in dataset:
            inputs = self.tokenizer(example["prompt"], return_tensors="pt", truncation=True, padding=True, max_length=512).to(self.device)
            total_prompt_tokens += inputs.input_ids.shape[1]
            output_tokens = self.model.generate(**inputs, max_length=1024, temperature=0.7, top_p=0.85, top_k=40, do_sample=True)
            total_generated += output_tokens.shape[1]
        return total_generated, total_prompt_tokens


class RougeEvaluator(ModelEvaluator):
    rouge_passed = None
    @time_execution
    def evaluate(self, dataset: Dataset, rouge_type: str, threshold: float, output_prefix: str):
        rouge = evaluate.load('rouge')
        prompts = [example["prompt"] for example in dataset]
        references = [example["completion"] for example in dataset]
        predictions = self.load_predictions()
        rouge_scores = rouge.compute(predictions=predictions, references=references, use_aggregator=False)[rouge_type]
        
        passed, failed = [], []
        for prompt, prediction, score in zip(prompts, predictions, rouge_scores):
            (passed if score > threshold else failed).append((prompt, prediction))
        self.rouge_passed = len(passed)

        # Save passed and failed to separate JSON files
        filename = "pretrained" + self.model_name if self.is_finetuned == False else "finetuned" + self.model_name
        self.save_to_json(passed, f"{filename}_rougepassed.json", output_prefix)
        self.save_to_json(failed, f"{filename}_rougefailed.json", output_prefix)
    
    def load_predictions(self):
        filename = f"./model_output/{self.prefix_file_name}_{self.model_name.split('/', 1)[0]}.json" if self.is_finetuned else f"./model_output/{self.model_name.split('/', 1)[0]}.json"
        with open(filename, 'r', encoding='utf-8') as file:
            return [entry["completion"] for entry in json.load(file)]

    def save_to_json(self, data, filename, output_prefix):
        """ Helper method to save data to a JSON file """
        with open(f"{output_prefix}{filename}", 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

