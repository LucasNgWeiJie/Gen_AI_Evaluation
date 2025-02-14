MODELS = ["stabilityai/stablelm-2-1_6b",
          "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
          "Qwen/Qwen1.5-1.8B-Chat"]
BASE_MODELS = [
    "StableLM 1.6B",
    "TinyLlama 1.1B",
    "Qwen 1.8B"
]
FINETUNED_MODELS = [
    "StableLM 1.6B (FT)",
    "TinyLlama 1.1B (FT)",
    "Qwen 1.8B (FT)"
]

DATASET_FILENAME = './dataset/filtered_all_prompts.json'
ASSERT_FILENAME = '../other_dataset/regex.json'
PREFIX = 'models/finetuned'
GRAPH_FILENAME_PREFIX = './eval_plots/'

