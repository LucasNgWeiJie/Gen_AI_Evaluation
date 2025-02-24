import json
import torch
import time
from datasets import load_dataset
from typing import Type

from model import *
from consts import *
from evaluation import *
from graph import*

class ModelManager():
    def __init__(self):
        self.model_list = MODELS
        self.pretrained_models = BASE_MODELS
        self.finetuned_models = FINETUNED_MODELS

    @time_execution
    def finetune_all_models(dataset_name: str):
        for model_name in MODELS:
            model_trainer = ModelTrainer(model_name, dataset_name)
            model_trainer.finetune_model()
        print("All Models created!")


class EvaluationManager():
    def __init__(self, option: str):
        self.model_list = MODELS
        self.model_type = option
        self.dataset = self.open_dataset_file(DATASET_FILENAME, True)
        self.times = []

    def graph_plot(self):
        '''Method to plot the graphs for the required evaluation'''
        print("Plotting graph....")

    def run_evaluation(self):
        '''Method to evaluate all models in model_list'''
        print("Running evaluation")

    def open_dataset_file(self, dataset_file: str, to_print: bool):
        if isinstance(dataset_file, str):
            data = load_dataset("json", data_files = dataset_file, split="train")
        else:
            data = dataset_file
        if to_print:
            print(f"{dataset_file} has {data.num_rows} prompt-completion pairs")
        return data
    
    def get_evaluators(self, model, evaluator_class: Type[ModelEvaluator]):
        """Returns a list of evaluators based on the selected model type"""
        if self.model_type == 'pretrained':
            return [evaluator_class(model, False)]
        elif self.model_type == 'finetuned':
            return [evaluator_class(model, True)]
        else:
            return [
                evaluator_class(model, False),
                evaluator_class(model, True)
            ]
    
class LossPerplexity(EvaluationManager):
    def __init__(self, option):
        super().__init__(option)
        self.losses = []
        self.perplexities = []
        self.run_evaluation()

    def run_evaluation(self):
        for model in self.model_list:
            evaluators = self.get_evaluators(model, LossPerplexityEvaluator)

            for evaluator in evaluators:
                time_taken = evaluator.evaluate(self.dataset)
                self.times.append(time_taken)
                self.losses.append(evaluator.avg_loss)
                self.perplexities.append(evaluator.perplexity)
    def graph_plot(self):
        return super().graph_plot()



class GenerateCompletion(EvaluationManager):   
    def __init__(self, option):
        super().__init__(option)
        self.time_completions = []
        self.time_prompt_tokens = []
        self.time_completion_tokens = []
        self.run_evaluation()

    def run_evaluation(self):
        for model in self.model_list:
            evaluators = self.get_evaluators(model, CompletionEvaluator)
            for evaluator in evaluators:
                evaluator.evaluate(self.dataset)
                self.time_completions.append(evaluator.time_completion)


    
    






