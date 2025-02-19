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

DATASET_FILENAME = './dataset/filtered_all_prompts.json'   # Path to dataset file
ASSERT_FILENAME = '../other_dataset/regex.json'   # Path to assert dataset file
PREFIX = 'models/finetuned' # Prefix for finetuned model
GRAPH_FILENAME_PREFIX = './eval_plots/'    # Path to save evaluation graph plots
ROUGE_THRESHOLD = 0.2 # Rouge threshold for rouge score
ROUGE_OUTPUT_PREFIX = './rouge_outputs/' # Path to save rouge outputs


