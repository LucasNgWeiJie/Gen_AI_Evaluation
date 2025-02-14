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
    # finetuned_times = []
    # pretrained_times = []

    if is_finetuned == "finetuned":
        # Calculates loss/perplexity for finetuned model
        for model in MODELS:
            finetuned_avg_loss, finetuned_perplexity = evaluate_loss_perplexity(model, data, True, PREFIX)
            finetuned_losses.append(finetuned_avg_loss)
            finetuned_perplexities.append(finetuned_perplexity)
        print(f"Perplexity Evaluation done\n Plotting graphs for finetuned models...")
        # plot graph
        plot_metrics_line(FINETUNED_MODELS,
                          finetuned_losses,
                          finetuned_perplexities,
                          GRAPH_FILENAME_PREFIX
                          + 'perplexity_loss_finetuned_metrics_line.png')
        plot_metrics_bar(FINETUNED_MODELS,
                         finetuned_losses,
                         finetuned_perplexities,
                         GRAPH_FILENAME_PREFIX
                         + 'perplexity_loss_finetuned_metrics_bar.png')
    elif is_finetuned == "pretrained":
        # Calculates loss/perplexity for pretrained model
        for model in MODELS:
            pretrained_avg_loss, pretrained_perplexity = evaluate_loss_perplexity(model, data, False, PREFIX)
            pretrained_losses.append(pretrained_avg_loss)
            pretrained_perplexities.append(pretrained_perplexity)
        # plot graph
        print(f"Perplexity Evaluation done\n Plotting graphs for pretrained models...")
        plot_metrics_line(BASE_MODELS,
                          pretrained_losses,
                          pretrained_perplexities,
                          GRAPH_FILENAME_PREFIX
                          + 'perplexity_loss_base_metrics_line.png')
        plot_metrics_bar(BASE_MODELS,
                         pretrained_losses,
                         pretrained_perplexities,
                         GRAPH_FILENAME_PREFIX
                         + 'perplexity_loss_base_metrics_bar.png')
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
        plot_metrics_bar(both_models,
                        both_losses,
                        both_perplexities,
                        GRAPH_FILENAME_PREFIX + "both_perplexity_loss_metrics_bar.png")
        


def evaluate_time_completion(data, is_finetuned: str):
    '''
    This function creates a graph to show the difference in time between
    pre-trained and finetuned
    '''
    print("---------------|Evaluating Time and creating completion dataset.....|-----------------------")
    finetuned_times = []
    pretrained_times = []
    print(f"Time evaluation done\n Plotting graph for dataset completion time...")
    if is_finetuned == "finetuned":
        for model in MODELS:
            finetuned_times.append(evaluate_time(model, data, True, PREFIX))
        print(f"Time evaluation done for finetuned models done\n Plotting graph for dataset completion time...")
        # plot graph for time
        plot_metrics_single(FINETUNED_MODELS,
                            'Times per prompt completion',
                            finetuned_times,
                            GRAPH_FILENAME_PREFIX + 'time_metrics.png')
    elif is_finetuned == "pretrained":
        for model in MODELS:
            pretrained_times.append(evaluate_time(model, data, False, PREFIX))
        print(f"Time evaluation done pretrained models done\n Plotting graph for dataset completion time...")
        # plot graph for time
        plot_metrics_single(BASE_MODELS,
                            'Times per prompt completion',
                            pretrained_times,
                            GRAPH_FILENAME_PREFIX + 'finetuned_time_metrics.png')
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
    pretrained_asserts = []
    
    if is_finetuned == "finetuned":
        for model in MODELS:
            finetuned_asserts.append(count_correct_assert_statements(model, True, PREFIX, ASSERT_FILENAME))
        
        print(f"Finetuned asserts evaluation done\n Plotting Graphs for assert metrics...")
        plot_metrics_single(FINETUNED_MODELS,
                            '# of Correct Asserts',
                            finetuned_asserts,
                            GRAPH_FILENAME_PREFIX +
                            'finetuned_assert_metrics.png')
    elif is_finetuned == "pretrained":
        for model in MODELS:
            pretrained_asserts.append(count_correct_assert_statements(model, False, PREFIX, ASSERT_FILENAME))

        print(f"Pretrained asserts evaluation done\n Plotting Graphs for assert metrics...")
        plot_metrics_single(BASE_MODELS,
                            '# of Correct Asserts',
                            pretrained_asserts,
                            GRAPH_FILENAME_PREFIX +
                            'assert_metrics.png')
    else:
        for model in MODELS:
            finetuned_asserts.append(count_correct_assert_statements(model, True, PREFIX, ASSERT_FILENAME))
            pretrained_asserts.append(count_correct_assert_statements(model, False, PREFIX, ASSERT_FILENAME))
        
        print(f"All asserts evaluation done\n Plotting Graphs for assert metrics...")
        # plot a comparison for both metrics
        both_models = BASE_MODELS + FINETUNED_MODELS
        both_asserts = pretrained_asserts + finetuned_asserts
        plot_metrics_single(both_models,
                            '# of correct asserts',
                            both_asserts,
                            GRAPH_FILENAME_PREFIX + 'both_assert_metrics.png')
        


def evaluate_all_rouge(data, is_finetuned: str):
    '''
    This function creates a graph to show the difference in rouge scores
    between pre-trained and finetuned
    '''
    # code to call evaluate_rouge on all models
    print("---------------|Evaluating Rouge score.....|-----------------------")
    finetuned_rouge1 = []
    finetuned_rouge2 = []
    finetuned_rougeL = []
    finetuned_rougeLsum = []
    
    pretrained_rouge1 = []
    pretrained_rouge2 = []
    pretrained_rougeL = []
    pretrained_rougeLsum = []

    # calculate for finetuned AND non pretrained
   
    # Plot graph for rouge score
    if is_finetuned == "finetuned":
        for model in MODELS:
            ft_r1, ft_r2, ft_rl, ft_rlsum = evaluate_rouge(model, data, True)
            finetuned_rouge1.append(ft_r1)
            finetuned_rouge2.append(ft_r2)
            finetuned_rougeL.append(ft_rl)
            finetuned_rougeLsum.append(ft_rlsum)
        # plot any of the 4 rouge score graphs for finetuned (edit the arguments)
        print(f"Rouge calculation done\n Plotting graphs for all Rouge scores...")
        plot_metrics_single(FINETUNED_MODELS,
                            "Finetuned Rouge LCS",
                            finetuned_rougeL,
                            GRAPH_FILENAME_PREFIX + 'finetuned_rougeLCS.png')
    elif is_finetuned == "pretrained":
        for model in MODELS:
            pt_r1, pt_r2, pt_rl, pt_rlsum = evaluate_rouge(model, data, False)
            pretrained_rouge1.append(pt_r1)
            pretrained_rouge2.append(pt_r2)
            pretrained_rougeL.append(pt_rl)
            pretrained_rougeLsum.append(pt_rlsum)
        # plot any of the 4 rouge score graphs for pretrained (edit the arguments)
        print(f"Rouge calculation done\n Plotting graphs for all Rouge scores...")
        plot_metrics_single(BASE_MODELS,
                            "Pretrained Rouge LCS",
                            pretrained_rougeL,
                            GRAPH_FILENAME_PREFIX + 'pretrained_rougeLCS.png')
    else:
        for model in MODELS:
            ft_r1, ft_r2, ft_rl, ft_rlsum = evaluate_rouge(model, data, True)
            finetuned_rouge1.append(ft_r1)
            finetuned_rouge2.append(ft_r2)
            finetuned_rougeL.append(ft_rl)
            finetuned_rougeLsum.append(ft_rlsum)

            pt_r1, pt_r2, pt_rl, pt_rlsum = evaluate_rouge(model, data, False)
            pretrained_rouge1.append(pt_r1)
            pretrained_rouge2.append(pt_r2)
            pretrained_rougeL.append(pt_rl)
            pretrained_rougeLsum.append(pt_rlsum)
        # plot both
        print(f"Rouge calculation done\n Plotting graphs for all Rouge scores...")
        both_models = BASE_MODELS + FINETUNED_MODELS
        both_rougeL = pretrained_rougeL + finetuned_rougeL
        plot_metrics_single(both_models,
                            "Comparison Rouge LCS",
                            both_rougeL,
                            GRAPH_FILENAME_PREFIX + 'both_rougeLCS.png')
    

def eval_suite(dataset_filename: str, is_finetuned: str):
    '''
    This function runs all the evaluation metrics
    and creates the relevant graphs
    '''
    start_time = time.time()
    print(f"---------------Starting Model Evaluation-----------------------")
    data = open_dataset_file(dataset_filename, True)
    evaluate_all_perplexity_loss(data, is_finetuned)
    evaluate_time_completion(data, is_finetuned)
    evaluate_all_asserts(is_finetuned)
    evaluate_all_rouge(data, is_finetuned)
    end_time = time.time()
    total_time = end_time - start_time
    print(f"---------------Evaluation Complete in {total_time} seconds-----------------------")