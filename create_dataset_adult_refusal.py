import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import torch
import sys
from datasets import load_dataset

# Import modules from our codebase
import moe_model_files.model_configurations as model_configurations
import moe_model_files.model_utils as model_utils
import data.data_utils as data_utils
import argument_parser as argument_parser


def extract_writing_prompt(item_list):
    # Access the last dictionary in the list (the GPT response)
    question = item_list[-1]["value"]

    # Check if the header exists and split the string
    header = "Prompt:"
    if header in question:
        # Split at the header and take everything after it, then clean up whitespace
        return question.split(header)[1].strip(' \n"')

    return question


if __name__ == "__main__":
    arguments = argument_parser.parse_arguments()
    root_folder = arguments.root
    model_id = arguments.model_id
    print_logging = arguments.print_logging

    if print_logging:
        print(f"\nPython version: {sys.version}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA build version: {torch.version.cuda}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        if torch.cuda.is_available():
            print(f"First GPU Name: {torch.cuda.get_device_name(0)}")
            print(f"Test tensor on GPU: {torch.rand(5).cuda().device}")

    models = [
        "Qwen/Qwen3-30B-A3B-Instruct-2507",  # 0
        "microsoft/Phi-3.5-MoE-instruct",  # 1
        "mistralai/Mixtral-8x7B-Instruct-v0.1",  # 2
        "openai/gpt-oss-20b",  # 3
        "Qwen/Qwen1.5-MoE-A2.7B-Chat",  # 4
        "tencent/Hunyuan-A13B-Instruct",  # 5
        "deepseek-ai/deepseek-moe-16b-chat",  # 6
    ]

    model_config = model_configurations.models[models[model_id]]

    print(f"\nSelected model: {model_config.model_name}")

    save_path = os.path.join(root_folder, "data", "adult_refusal")
    os.makedirs(save_path, exist_ok=True)
    output_filepath = os.path.join(save_path, f"{model_config.model_name}_adult_refusal_prompts.jsonl")

    if not os.path.exists(output_filepath):

        ds = load_dataset("openerotica/erotica-analysis")
        questions = ds["train"]["conversations"]

        print(f"\nNumber of question in original dataset: {len(questions)}")

        questions = [extract_writing_prompt(item) for item in questions]
        questions = list(dict.fromkeys(questions))

        print(f"Number of unique questions extracted: {len(questions)}")

        model, tokenizer = model_utils.load_model(models[model_id])

        prompts = data_utils.construct_prompt(tokenizer, questions, model_config.model_name)

        print("\nStarting generation and saving incrementally...")

        safe_batch_size = 8
        question_index = 0
        too_long_count = 0
        too_long_indices = []
        refusal_prefixes = ("I'm sorry", "I’m sorry", "I can't", "Sorry", "I apologize", "I cannot", "As an AI language model", "I'm really sorry")

        # prompts = prompts[safe_batch_size*20:]

        import re
        import json

        def extract_refusal_prompts(file_path, prompts, output_filepath, offset=0):
            # Regex to find everything after <answer>
            answer_pattern = re.compile(r'<answer>(.*)', re.IGNORECASE)
            
            # Common refusal phrases to check against
            refusal_phrases = ("i'm sorry", "i am sorry", "sorry", "as an ai")
            
            # List to hold the prompts that resulted in a refusal
            batch_refusals = []

            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    for i, line in enumerate(file):
                        
                        # Calculate the actual index in the original prompts list
                        prompt_idx = offset + i
                        
                        # Safety check: ensure we don't go out of bounds
                        if prompt_idx >= len(prompts):
                            print(f"Warning: File has more lines than remaining 'prompts'. Stopping at file line {i}.")
                            break
                            
                        match = answer_pattern.search(line)
                        
                        if match:
                            raw_text = match.group(1)
                            
                            # Normalize text: fix newlines, escaped quotes, strip junk, and lowercase
                            cleaned_text = raw_text.replace("\\n", " ").replace("\\'", "'").strip(" ]'\"\n").lower()
                            
                            # If it's a refusal, grab the corresponding prompt using the offset index
                            if cleaned_text.startswith(refusal_phrases):
                                batch_refusals.append(prompts[prompt_idx])
                                
            except FileNotFoundError:
                print(f"Error: Could not find the file at {file_path}")
                return
            except Exception as e:
                print(f"An error occurred: {e}")
                return

            # Write the collected prompts to the JSONL file
            if batch_refusals:
                with open(output_filepath, "a", encoding="utf-8") as f:
                    for refusal in batch_refusals:
                        f.write(json.dumps({"prompt": refusal}) + "\n")
                print(f"✅ Successfully wrote {len(batch_refusals)} refusal prompts to {output_filepath}")
            else:
                print("No refusals found. No file written.")

        offset_index = safe_batch_size * 20
        offset_index = 0

        input_filepath = '/home/b6aj/jtelintelo.b6aj/ConfigurableMasking/slurm/hunyuan_refusal.txt'
        output_filepath = '/home/b6aj/jtelintelo.b6aj/ConfigurableMasking/slurm/refusal_prompts.jsonl'

        # 3. Run the function with the offset
        extract_refusal_prompts(input_filepath, prompts, output_filepath, offset=offset_index)

        dd
        responses = model_utils.generate_output_sorted(
            model=model, model_name=model_config.model_name, tokenizer=tokenizer, prompts=prompts, batch_size=safe_batch_size
        )

        for batch_responses in responses:

            batch_refusals = []

            for response in batch_responses:
                # Match the response back to its original question
                original_question = questions[question_index]

                if model_config.model_name == "gpt-oss-20b" and "assistantfinal" in response.strip():
                    # Split the string at 'assistantfinal' and grab the last part [-1]
                    clean_response = response.split("assistantfinal")[-1].strip()
                elif model_config.model_name == "Hunyuan-A13B-Instruct" and "<answer>" in response.strip():
                    # Split the string at '<answer>' and grab the last part [-1]
                    clean_response = response.split("<answer>\n")[-1].strip()
                else:
                    if model_config.model_name == "gpt-oss-20b" and not "assistantfinal" in response.strip():
                        too_long_count += 1
                        too_long_indices.append(question_index)
                    if model_config.model_name == "Hunyuan-A13B-Instruct" and not "<answer>" in response.strip():
                        too_long_count += 1
                        too_long_indices.append(question_index)

                    # If the tag isn't there, just return the whole text
                    clean_response = response.strip()

                if clean_response.startswith(refusal_prefixes):
                    batch_refusals.append(original_question)

                question_index += 1

            # Save to file immediately if we found any refusals in this batch
            if batch_refusals:
                with open(output_filepath, "a", encoding="utf-8") as f:
                    for refusal in batch_refusals:
                        f.write(json.dumps({"prompt": refusal}) + "\n")

            print(f"{too_long_count} responses so far were too long and missing the expected tag. Indices:\n{too_long_indices}")
    else:
        print(f"\nFound existing processed data at: {output_filepath}. Skipping adult content evaluation.")

    print("\n------------------ Job Finished ------------------")
