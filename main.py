'''
Main function to run all the evaluation metric
'''
from utils import *
from consts import *

# Initial run creates the models
# create_all_models(DATASET_FILENAME)

# Evaluates all metrics regardless, second arg controls graph output
eval_suite(DATASET_FILENAME, "finetuned")
eval_suite(DATASET_FILENAME, "pretrained")
eval_suite(DATASET_FILENAME, "both")