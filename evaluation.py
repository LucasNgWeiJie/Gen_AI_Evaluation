'''
This class defines the evaluation of models
'''
import gc
import json
import time
import re
import torch
import evaluate

from datasets import (Dataset, DatasetDict, IterableDataset,
                      IterableDatasetDict, load_dataset)
from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer


def evaluate_loss_perplexity(model_name: str,
                             dataset: str | (DatasetDict |
                                                  Dataset |
                                                  IterableDatasetDict |
                                                  IterableDataset),
                             is_finetuned: bool,
                             prefix_file_name: str = 'finetuned'):
    '''
    This function takes in the model name and dataset
    Outputs the average loss and perplexity of the model
    '''
    # for time recording purposes
    start_time = time.time()
    if is_finetuned:
        print(f"Evaluatating loss and perplexity of finetuned {model_name}.........")
    else:
        print(f"Evaluatating loss and perplexity of pretrained {model_name}.........")
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load model
    if is_finetuned:
        model_name = f'./{prefix_file_name}_{model_name}'
        evl_model = AutoPeftModelForCausalLM.from_pretrained(
            model_name).to(device)
    else:
        evl_model = AutoModelForCausalLM.from_pretrained(
            model_name).to(device)
        
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    evl_model.eval()
    total_loss = 0
    total_tokens = 0
    for example in dataset:
        inputs = tokenizer(example["prompt"],
                           return_tensors="pt",
                           truncation=True,
                           padding=True,
                           max_length=512).to(device)
        targets = tokenizer(example["completion"],
                            return_tensors="pt",
                            truncation=True,
                            padding=True,
                            max_length=512).input_ids.to(device)
        labels = torch.cat([inputs.input_ids, targets], dim=-1).to(device).long()
        with torch.no_grad():
            outputs = evl_model(input_ids=labels, labels=labels)
            loss = outputs.loss * labels.size(1)
            total_loss += loss.item()
            total_tokens += labels.size(1)
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss))

    end_time = time.time()
    time_taken = end_time - start_time
    print(f"{model_name} evaluation completed in {time_taken} seconds")
    del evl_model
    del tokenizer
    del dataset
    torch.cuda.empty_cache()
    gc.collect()
    return avg_loss, perplexity


def evaluate_time(model_name: str,
                  dataset: str | (
                      DatasetDict | Dataset |
                      IterableDatasetDict | IterableDataset
                  ),
                  is_finetuned: bool,
                  prefix_file_name: str = 'finetuned'
                  ):
    '''
    This function evaluates how long it takes for each prediction on average
    and creates the completion dataset for each model.
    It returns:
    - average time per prompt-completion
    - average time per completion token
    - average time per prompt token
    '''
    if is_finetuned:
        print(f"Evaluating time taken for finetuned {model_name}")
    else:
        print(f"Evaluating time taken for pretrained {model_name}")
    
    start_time = time.time()
    total_tokens_generated, total_prompt_tokens = create_completion_dataset(
        model_name, dataset, is_finetuned, prefix_file_name
    )
    end_time = time.time()

    total_time = end_time - start_time
    time_per_prompt_completion = total_time / dataset.num_rows
    time_per_prompt_token = total_time / total_prompt_tokens if total_prompt_tokens > 0 else 0
    time_per_completion_token = total_time / total_tokens_generated if total_tokens_generated > 0 else 0
    print(f"{model_name} time metrics:")
    print(f"Each prompt-completion takes on average: {time_per_prompt_completion} seconds")
    print(f"Each prompt token takes on average: {time_per_prompt_token} seconds")
    print(f"Each completion token takes on average: {time_per_completion_token} seconds")
    
    return time_per_prompt_completion


def create_completion_dataset(model_name: str,
                              dataset: str | (
                                  DatasetDict | Dataset |
                                  IterableDatasetDict | IterableDataset
                                  ),
                              is_finetuned: bool,
                              prefix_file_name: str = 'finetuned'):
    '''
    This function generates the responses (completions) to the prompts in the dataset
    It will return:
    - total tokens generated (for completions)
    - total number of prompt tokens
    - total number of completions (to calculate the time per completion)
    '''
    file_name = model_name
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    if is_finetuned:
        model_name = f'./{prefix_file_name}_{model_name}'
        file_name = "finetuned_" + file_name
        evl_model = AutoPeftModelForCausalLM.from_pretrained(model_name).to(device)
    else:
        evl_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    
    print(f"Writing output to {file_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize counters
    total_tokens_generated = 0
    total_prompt_tokens = 0
    
    res = []

    # Process dataset
    for example in dataset:
        # Tokenize prompt
        inputs = tokenizer(example["prompt"],
                           return_tensors="pt",
                           truncation=True,
                           padding=True,
                           max_length=512).to("cuda")
        
        # Count prompt tokens
        prompt_tokens = len(inputs.input_ids[0])
        total_prompt_tokens += prompt_tokens

        # Generate output
        output_tokens = evl_model.generate(
            **inputs,
            pad_token_id=tokenizer.pad_token_id,
            max_length=2048,
            temperature=0.5,
            top_p=0.9,
            top_k=50,
            do_sample=True
        )
        
        # Count completion tokens
        completion_tokens = len(output_tokens[0])
        total_tokens_generated += completion_tokens

        # Decode and store result
        completed = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        res.append({
            "prompt": example["prompt"],
            "completion": completed
        })
    
    # Write results to a file
    with open(f"./model_output/{file_name.split('/')[0]}.json", 'w', encoding='utf-8') as file:
        json.dump(res, file, indent=4)
    
    return total_tokens_generated, total_prompt_tokens


def count_correct_assert_statements(model_name: str,
                                    is_finetuned: bool,
                                    prefix_file_name: str,
                                    assert_file_name: str):
    '''
    This function calculates the total number of assert statements
    '''
    if is_finetuned:
        print(f"Evaluating asserts for finetuned {model_name}")
    else:
        print(f"Evaluating asserts for pretrained {model_name}")
    start_time = time.time()
    file_name = model_name
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load model
    if is_finetuned:
        model_name = f'./{prefix_file_name}_{model_name}'
        file_name = "finetuned_" + file_name
        evl_model = AutoPeftModelForCausalLM.from_pretrained(
            model_name).to(device)
    else:
        evl_model = AutoModelForCausalLM.from_pretrained(
            model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    # load assert statements
    with open(assert_file_name, 'r', encoding='utf-8') as file:
        data = json.load(file)
    count_correct = 0
    '''
    For each test in the loaded data (each test contains a prompt and expected responses):
    The prompt is tokenized using the model’s tokenizer.
    The model generates a sequence of tokens as output based on the prompt.
    The generated tokens are decoded back into text using the tokenizer.
    For each expected completion (test["test"]), the function checks if it exists in the model's generated output (completed).
    It uses regular expressions (re.search()) to match the expected output with the generated one.
    If a match is found, count_correct is incremented.
    '''
    for test in data:
        inputs = tokenizer(test["prompt"],
                           return_tensors="pt",
                           truncation=True,
                           padding=True,
                           max_length=512).to("cuda")
        attention_mask = inputs["attention_mask"]
        output_tokens = evl_model.generate(
            inputs.input_ids,
            attention_mask=attention_mask,
            temperature=0.5,
            top_p=0.9,
            top_k=50,
            do_sample=True
        )
        completed = tokenizer.decode(output_tokens[0],
                                     skip_special_tokens=True)
        for exp in test["test"]:
            if re.search(exp, completed):
                count_correct += 1
    end_time = time.time()
    time_taken = end_time - start_time
    print(f"Assert statements counted for {model_name} in {time_taken} seconds")
    return count_correct

def evaluate_rouge(model_name: str,
                    dataset: str | (DatasetDict |
                                        Dataset |
                                        IterableDatasetDict |
                                        IterableDataset),
                   is_finetuned: bool):
   
    '''
    Calculates the 4 different rouge scores of a model. 
    Returns them as separate variables
    '''
    if is_finetuned:
        print(f"Evaluating Rouge Scores for finetuned {model_name}")
    else:
        print(f"Evaluating Rouge Scores for pretrained {model_name}")
    start_time = time.time()
    rouge = evaluate.load('rouge')
    references = [r["completion"] for r in dataset]
    
    # opens file, location is different between finetuned and pretrained
    if is_finetuned:
        finetuned_model_path = (
            f"./model_output/finetuned_{model_name.split('/', maxsplit=-1)[0]}.json"
        )
        with open(finetuned_model_path, 'r', encoding='utf-8') as file:
            predictions = json.load(file)
        predictions = [r["completion"] for r in predictions]
    else:
        with open(f"./model_output/{model_name.split('/', maxsplit=-1)[0]}.json",
                    'r',
                    encoding='utf-8') as file:
            predictions = json.load(file)
        predictions = [r["completion"] for r in predictions]
    # Rouge score calculations
    rouge1 = rouge.compute(predictions=predictions,
                                references=references)["rouge1"]
    rouge2 = rouge.compute(predictions=predictions,
                                references=references)["rouge2"]
    rougeL = rouge.compute(predictions=predictions,
                                references=references)["rougeL"]
    rougeLsum = rouge.compute(predictions=predictions,
                                references=references)["rougeLsum"]
    end_time = time.time()
    time_taken = end_time - start_time
    print(f"All rouge scores computed for {model_name} in {time_taken} seconds")

    return rouge1, rouge2, rougeL, rougeLsum
