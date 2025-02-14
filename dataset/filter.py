import json
import random

def get_random_prompts(input_file, output_file, num_prompts):
    # Load data from the input file
    with open(input_file, 'r') as infile:
        data = json.load(infile)
    
    # Check if there are enough prompt-completion pairs
    if len(data) < num_prompts:
        raise ValueError(f"Not enough prompt-completion pairs in the input file. Requested: {num_prompts}, Available: {len(data)}")
    
    # Randomly select num_prompts pairs
    selected_prompts = random.sample(data, num_prompts)
    
    # Write the selected pairs to the output file
    with open(output_file, 'w') as outfile:
        json.dump(selected_prompts, outfile, indent=4)

def main():
    input_file = 'all_prompts_uicc.json'  
    output_file = 'filtered_prompts.json'  
    num_prompts = 5
    
    get_random_prompts(input_file, output_file, num_prompts)

if __name__ == '__main__':
    main()

