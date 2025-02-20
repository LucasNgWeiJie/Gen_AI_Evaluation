import json
import evaluate
import torch
import gc
import time
from datasets import load_dataset

from evaluation import (evaluate_loss_perplexity,
                        evaluate_time,
                        count_correct_assert_statements,
                        evaluate_rouge)
from graph import (plot_metrics_line, 
                    plot_metrics_single,
                    plot_metrics_bar)
from model import *
from consts import *

def create_all_models(dataset_name: str):
    '''
    This function is used to create all models in the constants
    '''
    for model in MODELS:
        try:
            create_model(model, dataset_name, PREFIX)
        except (OSError, ValueError, RuntimeError) as e:
            print(f'Error in creating model {model}: {e}')
            continue

def open_dataset_file(dataset_file: str, to_print: bool):
    if isinstance(dataset_file, str):
        data = load_dataset("json", data_files=dataset_file, split="train")
    else:
        data = dataset_file
    if to_print:
        print(f"{dataset_file} has {data.num_rows} of prompt-completion pairs")
    return data

def evaluate_all_perplexity_loss(data, is_finetuned: str):
    '''
    This function creates a graph to show the difference in metric between
    pre-trained and finetuned perplexity and loss
    '''
    print("---------------|Evaluating Perplexity and Loss....|-----------------------")
    finetuned_losses = []
    pretrained_losses = []
    finetuned_perplexities = []
    pretrained_perplexities = []
    finetuned_times = []
    pretrained_times = []

    if is_finetuned == "finetuned":
        # Calculates loss/perplexity for finetuned model
        for model in MODELS:
            finetuned_avg_loss, finetuned_perplexity, finetuned_time = evaluate_loss_perplexity(model, data, True, PREFIX)
            finetuned_losses.append(finetuned_avg_loss)
            finetuned_perplexities.append(finetuned_perplexity)
            finetuned_times.append(finetuned_time)
        print(f"Perplexity Evaluation done\n Plotting graphs for finetuned models...")
        # plot graph
        plot_metrics_bar(FINETUNED_MODELS,
                         finetuned_losses,
                         finetuned_perplexities,
                         GRAPH_FILENAME_PREFIX
                         + 'perplexity_loss_finetuned_metrics_bar.png')
        plot_metrics_single(FINETUNED_MODELS,
                            'Time for loss/perplexity calculation',
                            finetuned_times,
                            GRAPH_FILENAME_PREFIX + 'finetuned_lossper_time.png')
        
    elif is_finetuned == "pretrained":
        # Calculates loss/perplexity for pretrained model
        for model in MODELS:
            pretrained_avg_loss, pretrained_perplexity, pretrained_time = evaluate_loss_perplexity(model, data, False, PREFIX)
            pretrained_losses.append(pretrained_avg_loss)
            pretrained_perplexities.append(pretrained_perplexity)
            pretrained_times.append(pretrained_time)
        # plot graph
        print(f"Perplexity Evaluation done\n Plotting graphs for pretrained models...")
        plot_metrics_bar(BASE_MODELS,
                         pretrained_losses,
                         pretrained_perplexities,
                         GRAPH_FILENAME_PREFIX
                         + 'perplexity_loss_base_metrics_bar.png')
        plot_metrics_single(BASE_MODELS,    
                            'Time for loss/perplexity calculation',
                            pretrained_times,
                            GRAPH_FILENAME_PREFIX + 'pretrained_lossper_time.png')
    else:
        for model in MODELS:
            pretrained_avg_loss, pretrained_perplexity = evaluate_loss_perplexity(model, data, False, PREFIX)
            pretrained_losses.append(pretrained_avg_loss)
            pretrained_perplexities.append(pretrained_perplexity)
            finetuned_avg_loss, finetuned_perplexity = evaluate_loss_perplexity(model, data, True, PREFIX)
            finetuned_losses.append(finetuned_avg_loss)
            finetuned_perplexities.append(finetuned_perplexity)
        print(f"Perplexity Evaluation done\n Plotting graphs for all...")
        # plot graph that compares both
        both_models = BASE_MODELS + FINETUNED_MODELS
        both_losses = pretrained_losses + finetuned_losses
        both_perplexities = pretrained_perplexities + finetuned_perplexities
        both_times = finetuned_times + pretrained_times
        plot_metrics_bar(both_models,
                        both_losses,
                        both_perplexities,
                        GRAPH_FILENAME_PREFIX + "both_perplexity_loss_metrics_bar.png")
        plot_metrics_single(both_models,
                            'Time for loss/perplexity calculation',
                            both_times,
                            GRAPH_FILENAME_PREFIX + 'both_lossper_time.png')
    
        


def evaluate_time_completion(data, is_finetuned: str):
    '''
    This function creates a graph to show the difference in time between
    pre-trained and finetuned
    '''
    print("---------------|Evaluating Time and creating completion dataset.....|-----------------------")
    finetuned_times = []
    pretrained_times = []
    if is_finetuned == "finetuned":
        for model in MODELS:
            finetuned_times.append(evaluate_time(model, data, True, PREFIX))
        print(f"Time evaluation done for finetuned models done\n Plotting graph for dataset completion time...")
        # plot graph for time
        plot_metrics_single(FINETUNED_MODELS,
                            'Times per prompt completion',
                            finetuned_times,
                            GRAPH_FILENAME_PREFIX + 'finetuned_time_metrics.png')
    elif is_finetuned == "pretrained":
        for model in MODELS:
            pretrained_times.append(evaluate_time(model, data, False, PREFIX))
        print(f"Time evaluation done pretrained models done\n Plotting graph for dataset completion time...")
        # plot graph for time
        plot_metrics_single(BASE_MODELS,
                            'Times per prompt completion',
                            pretrained_times,
                            GRAPH_FILENAME_PREFIX + 'pretrained_time_metrics.png')
    else:
        # plot metrics that measure and compare time for both
        for model in MODELS:
            finetuned_times.append(evaluate_time(model, data, True, PREFIX))
            pretrained_times.append(evaluate_time(model, data, False, PREFIX))
        print(f"Time evaluation done for all models done\n Plotting graph for dataset completion time...")
        # plot graph for time
        both_models = BASE_MODELS + FINETUNED_MODELS
        both_times = pretrained_times + finetuned_times
        plot_metrics_single(both_models,
                            'Times per prompt completion',
                            both_times,
                            GRAPH_FILENAME_PREFIX + 'both_time_metrics.png')


def evaluate_all_asserts(is_finetuned: str):
    '''
    This function creates a graph to show the difference in asserts between
    pre-trained and finetuned
    '''
    print("---------------|Evaluating Asserts........|-----------------------")
    finetuned_asserts = []
    finetuned_times = []
    pretrained_asserts = []
    pretrained_times = []
    if is_finetuned == "finetuned":
        for model in MODELS:
            finetuned_assert, finetuned_time = count_correct_assert_statements(model, True, PREFIX, ASSERT_FILENAME)
            finetuned_asserts.append(finetuned_assert)
            finetuned_times.append(finetuned_time)
        print(f"Finetuned asserts evaluation done\n Plotting Graphs for assert metrics...")
        plot_metrics_single(FINETUNED_MODELS,
                            '# of Correct Asserts',
                            finetuned_asserts,
                            GRAPH_FILENAME_PREFIX +
                            'finetuned_assert_metrics.png')
        plot_metrics_single(FINETUNED_MODELS,
                            "Time taken for assert evaluation",
                            finetuned_times,
                            GRAPH_FILENAME_PREFIX + "Finetuned_asserts_time.png")
    elif is_finetuned == "pretrained":
        for model in MODELS:
            pretrained_assert, pretrained_time = count_correct_assert_statements(model, True, PREFIX, ASSERT_FILENAME)
            pretrained_asserts.append(pretrained_assert)
            pretrained_times.append(pretrained_time)

        print(f"Pretrained asserts evaluation done\n Plotting Graphs for assert metrics...")
        plot_metrics_single(BASE_MODELS,
                            '# of Correct Asserts',
                            pretrained_asserts,
                            GRAPH_FILENAME_PREFIX +
                            'assert_metrics.png')
        plot_metrics_single(BASE_MODELS,
                            "Time taken for assert evaluation",
                            pretrained_times,
                            GRAPH_FILENAME_PREFIX + "pretrained_asserts_time.png")
    else:
        for model in MODELS:
            finetuned_assert, finetuned_time = count_correct_assert_statements(model, True, PREFIX, ASSERT_FILENAME)
            finetuned_asserts.append(finetuned_assert)
            finetuned_times.append(finetuned_time)
            pretrained_assert, pretrained_time = count_correct_assert_statements(model, True, PREFIX, ASSERT_FILENAME)
            pretrained_asserts.append(pretrained_assert)
            pretrained_times.append(pretrained_time)
        
        print(f"All asserts evaluation done\n Plotting Graphs for assert metrics...")
        # plot a comparison for both metrics
        both_models = BASE_MODELS + FINETUNED_MODELS
        both_asserts = pretrained_asserts + finetuned_asserts
        both_times = pretrained_times + finetuned_times
        plot_metrics_single(both_models,
                            '# of correct asserts',
                            both_asserts,
                            GRAPH_FILENAME_PREFIX + 'both_assert_metrics.png')
        plot_metrics_single(both_models,
                            "Time taken for assert evaluation",
                            both_times,
                            GRAPH_FILENAME_PREFIX + 'both_asserts_time.png')
        

def evaluate_all_rouge(data, is_finetuned: str, rouge_type: str, threshold: int):
    '''
    This function creates a graph to show the difference in rouge scores
    between pre-trained and finetuned
    '''
    # code to call evaluate_rouge on all models
    print("---------------|Evaluating Rouge score.....|-----------------------")
    finetuned_rouge_scores = []
    finetuned_times = []
    
    pretrained_rouge_scores = []
    pretrained_times = []
   
    if is_finetuned == "finetuned":   
        for model in MODELS:
            finetuned_rouge_score, passed_prompts, passed_completions, failed_prompts, failed_completions, finetuned_time = evaluate_rouge(model, data, True, rouge_type, threshold)
            finetuned_rouge_scores.append(finetuned_rouge_score)
            finetuned_times.append(finetuned_time)
            print(f"-------------------------------------Writing outputs that failed rouge score{is_finetuned}_{model}-------------------------------------------")
            # Create a list of dictionaries with prompt and completion pairs
            passed_outputs = [{"prompt": prompt, "completion": completion} for prompt, completion in zip(passed_prompts, passed_completions)]
            failed_outputs = [{"prompt": prompt, "completion": completion} for prompt, completion in zip(failed_prompts, failed_completions)]
            # Write the data to a JSON file
            with open(ROUGE_OUTPUT_PREFIX + f'finetuned_rougepassed_{model.split("/")[0]}.json', 'w') as json_file:
                json.dump(passed_outputs, json_file, indent=4)
            with open(ROUGE_OUTPUT_PREFIX + f'finetuned_rougefailed_{model.split("/")[0]}.json', 'w') as json_file:
                json.dump(failed_outputs, json_file, indent=4)
        # plot graph for finetuned models
        print(f"Rouge calculation done\n Plotting graphs for all Rouge scores...")
        plot_metrics_single(FINETUNED_MODELS,
                            "Finetued Rouge LCS Scores",
                            finetuned_rouge_scores,
                            GRAPH_FILENAME_PREFIX + 'finetuned_rougeLCS.png')
        plot_metrics_single(FINETUNED_MODELS,
                            "Time Taken for rouge evaluation",
                            finetuned_times,
                            GRAPH_FILENAME_PREFIX + 'finetuned_rouge_times.png')

    elif is_finetuned == "pretrained":
        for model in MODELS:
            pretrained_rouge_score, failed_prompts, failed_completions, pretrained_time = evaluate_rouge(model, data, False, rouge_type, threshold)
            pretrained_rouge_scores.append(pretrained_rouge_score)
            pretrained_times.append(pretrained_time)
            print(f"-------------------------------------Writing outputs that failed rouge score{is_finetuned}_{model}-------------------------------------------")
            # Create a list of dictionaries with prompt and completion pairs
            passed_outputs = [{"prompt": prompt, "completion": completion} for prompt, completion in zip(passed_prompts, passed_completions)]
            failed_outputs = [{"prompt": prompt, "completion": completion} for prompt, completion in zip(failed_prompts, failed_completions)]
            # Write the data to a JSON file
            with open(ROUGE_OUTPUT_PREFIX + f'pretrained_rougepassed_{model.split("/")[0]}.json', 'w') as json_file:
                json.dump(passed_outputs, json_file, indent=4)
            with open(ROUGE_OUTPUT_PREFIX + f'pretrained_rougefailed_{model.split("/")[0]}.json', 'w') as json_file:
                json.dump(failed_outputs, json_file, indent=4)
        # plot for pretrained
        print(f"Rouge calculation done\n Plotting graphs for all Rouge scores...")
        plot_metrics_single(BASE_MODELS,
                            "Pretrained Rouge LCS scores",
                            pretrained_rouge_scores,
                            GRAPH_FILENAME_PREFIX + 'pretrained_rougeLCS.png')
        plot_metrics_single(BASE_MODELS,
                            "Time Taken for rouge evaluation",
                            pretrained_times,
                            GRAPH_FILENAME_PREFIX + 'pretrained_rouge_times.png')

    else:
        for model in MODELS:
            finetuned_rouge_score, passed_prompts_ft, passed_completions_ft, failed_prompts_ft, failed_completions_ft, finetuned_time = evaluate_rouge(model, data, True, rouge_type, threshold)
            finetuned_rouge_scores.append(finetuned_rouge_score)
            finetuned_times.append(finetuned_time)

            pretrained_rouge_score, passed_prompts_pt, passed_completions_pt, failed_prompts_pt, failed_completions_pt, pretrained_time = evaluate_rouge(model, data, False, rouge_type, threshold)
            pretrained_rouge_scores.append(pretrained_rouge_score)
            pretrained_times.append(pretrained_time)

            print(f"-------------------------------------Writing outputs that passed/failed rouge score {is_finetuned}_{model}-------------------------------------------")
            # writ for pretrained
            passed_outputs_pt = [{"prompt": prompt, "completion": completion} for prompt, completion in zip(passed_prompts_pt, passed_completions_pt)]
            failed_outputs_pt = [{"prompt": prompt, "completion": completion} for prompt, completion in zip(failed_prompts_pt, failed_completions_pt)]
            # Write the data to a JSON file
            with open(ROUGE_OUTPUT_PREFIX + f'pretrained_rougepassed_{model.split("/")[0]}.json', 'w') as json_file:
                json.dump(passed_outputs_pt, json_file, indent=4)
            with open(ROUGE_OUTPUT_PREFIX + f'pretrained_rougefailed_{model.split("/")[0]}.json', 'w') as json_file:
                json.dump(failed_outputs_pt, json_file, indent=4)    

            # write for finetuned  
            passed_outputs_ft = [{"prompt": prompt, "completion": completion} for prompt, completion in zip(passed_prompts_ft, passed_completions_ft)]
            failed_outputs_ft = [{"prompt": prompt, "completion": completion} for prompt, completion in zip(failed_prompts_ft, failed_completions_ft)]
            # Write the data to a JSON file
            with open(ROUGE_OUTPUT_PREFIX + f'finetuned_rougepassed_{model.split("/")[0]}.json', 'w') as json_file:
                json.dump(passed_outputs_ft, json_file, indent=4)
            with open(ROUGE_OUTPUT_PREFIX + f'finetuned_rougefailed_{model.split("/")[0]}.json', 'w') as json_file:
                json.dump(failed_outputs_ft, json_file, indent=4)           

        # plot both
        print(f"Rouge calculation done\n Plotting graphs for all Rouge scores...")
        both_models = BASE_MODELS + FINETUNED_MODELS
        both_rouge_scores = pretrained_rouge_scores + finetuned_rouge_scores
        both_times = finetuned_times + pretrained_times
        plot_metrics_single(both_models,
                            "Comparison Rouge LCS",
                            both_rouge_scores,
                            GRAPH_FILENAME_PREFIX + 'both_rougeLCS.png')
        plot_metrics_single(both_models,
                            "Time taken for rouge evaluation",
                            both_times,
                            GRAPH_FILENAME_PREFIX + 'both_rouge_times.png')
    

def eval_suite(dataset_filename: str, is_finetuned: str):
    '''
    This function runs all the evaluation metrics
    and creates the relevant graphs
    '''
    start_time = time.time()
    print(f"---------------Starting Model Evaluation-----------------------")
    data = open_dataset_file(dataset_filename, True)
    # evaluate_all_perplexity_loss(data, is_finetuned)
    # evaluate_time_completion(data, is_finetuned)
    # evaluate_all_asserts(is_finetuned)
    evaluate_all_rouge(data, is_finetuned, 'rougeL', ROUGE_THRESHOLD) # 'rouge1', 'rouge2', 'rougeL', 'rougeLsum'
    end_time = time.time()
    total_time = end_time - start_time
    print(f"---------------Evaluation Complete in {total_time} seconds-----------------------")