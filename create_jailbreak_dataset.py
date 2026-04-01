import moe_model_files.model_configurations as model_configurations
import moe_model_files.model_utils as model_utils
import data.data_utils as data_utils
import argument_parser as argument_parser

import json
import torch
import sys
import os
import pandas as pd
from tqdm import tqdm

# ====================================================================
# 1. Evaluation & Data Processing Functions
# ====================================================================
class SafetyEvaluator:
    def __init__(self, j_model, j_tokenizer):
        self.model = j_model
        self.tokenizer = j_tokenizer

    def evaluate(self, user_prompt, generated_text):
        # Construct the prompt for the judge model
        judge_prompt = data_utils.construct_judge_prompt(user_prompt, generated_text)

        # Call the moderate function from your model_utils
        judge_output = model_utils.moderate(self.model, self.tokenizer, judge_prompt)

        # Evaluate if the output was flagged as unsafe
        return "unsafe" in judge_output.lower()


def find_first_jailbreak_free_generation(conversation_entry, model, tokenizer, safety_evaluator, model_name):
    messages = conversation_entry['messages']
    current_context = []
    base_model_name = model_name.split("/")[-1]
    
    for i, msg in enumerate(messages):
        # Append current message to context
        current_context.append({"role": msg['role'], "content": msg['body']})
        
        if msg['role'] == 'user':
            
            # 1. Format context (DeepSeek system prompt exclusion)
            chat = [m for m in current_context if m["role"] != "system"] if base_model_name == "deepseek-moe-16b-chat" else current_context
                
            # 2. Apply chat template
            if base_model_name.startswith("Qwen3") or base_model_name == "Hunyuan-A13B-Instruct":
                prompt_str = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            else:
                prompt_str = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            
            # 3. Call external generation module (Batch size 1 to maintain isolated context)
            responses = model_utils.generate_output(
                model=model,
                model_name=base_model_name,
                tokenizer=tokenizer,
                prompts=[prompt_str],
                batch_size=1 
            )
            
            generated_text = responses[0]
            
            # 4. Evaluate safety
            is_harmful = safety_evaluator.evaluate(msg['body'], generated_text)
            
            if is_harmful:
                print(f"Jailbreak triggered at turn {i} for Question ID: {conversation_entry.get('question_id')}!")
                
                # Return the conversation history up to this point. 
                # `current_context` currently ends with the User prompt that caused the jailbreak.
                # This is perfect for feeding into your next script to gather the logits.
                return {
                    "question_id": conversation_entry.get('question_id'),
                    "tactic": conversation_entry.get('tactic'),
                    "jailbreak_turn": i,
                    "conversation_history": current_context.copy(), 
                    "jailbreak_response": generated_text # Saved for reference
                }
            else:
                # If safe, append the model's response and continue the conversation
                current_context.append({"role": "assistant", "content": generated_text})
                
    return None

def process_jailbreak_data(csv_path, jsonl_path):
    print(f"Loading data from: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: Could not find {csv_path}. Please ensure the file is in that folder.")
        return

    conversations = []

    for _, row in df.iterrows():
        turns = []

        for i in range(101):
            col = f"message_{i}"
            if col in df.columns:
                val = row[col]
                if pd.notna(val) and str(val).strip() != "":
                    try:
                        msg_data = json.loads(val)
                        turns.append(msg_data)
                    except json.JSONDecodeError:
                        continue

        if turns:
            entry = {
                "question_id": int(row["question_id"]) if "question_id" in row else None,
                "tactic": row.get("tactic", "Unknown"),
                "messages": turns,
            }
            conversations.append(entry)

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for conv in conversations:
            f.write(json.dumps(conv) + "\n")

    print(f"Extraction complete! Saved {len(conversations)} entries to: {jsonl_path}")


# ====================================================================
# 2. Main Execution
# ====================================================================
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
        "IntervitensInc/pangu-pro-moe-model",  # 7
    ]

    model_config = model_configurations.models[models[model_id]]
    print(f"\nSelected target model: {model_config.model_name}")

    # Define file paths
    data_dir = os.path.join(root_folder, "data", "jailbreak")
    os.makedirs(data_dir, exist_ok=True)
    input_csv_path = os.path.join(data_dir, "harmbench_behaviors.csv")
    output_jsonl_path = os.path.join(data_dir, "mhj_processed.jsonl")

    # Run the extraction from CSV to JSONL
    process_jailbreak_data(input_csv_path, output_jsonl_path)

    # ---------------------------------------------------------
    # Load Models (Target MoE and Judge)
    # ---------------------------------------------------------
    print("\nLoading Target MoE Model and Tokenizer...")
    model, tokenizer = model_utils.load_model(model_config.model_name)
    model.eval()
    
    # Note: Hook registration completely removed from here.

    print("\nLoading Judge Model (Safety Evaluator)...")
    judge_model_name = "meta-llama/Llama-Guard-3-8B"  
    judge_model, judge_tokenizer = model_utils.load_model(judge_model_name)
    judge_model.eval()

    safety_evaluator = SafetyEvaluator(judge_model, judge_tokenizer)

    # ---------------------------------------------------------
    # Load the Processed JSONL Data
    # ---------------------------------------------------------
    print(f"\nLoading conversation contexts from {output_jsonl_path}...")
    with open(output_jsonl_path, "r", encoding="utf-8") as f:
        conversations = [json.loads(line) for line in f]

    # ---------------------------------------------------------
    # Execute Multi-Turn Jailbreak Search
    # ---------------------------------------------------------
    print(f"\nStarting Free-Generation evaluation on {len(conversations)} multi-turn sequences...")
    successful_jailbreaks = []

    for conv_entry in tqdm(conversations, desc="Evaluating Conversations"):
        result = find_first_jailbreak_free_generation(
            conversation_entry=conv_entry, 
            model=model, 
            tokenizer=tokenizer, 
            safety_evaluator=safety_evaluator, 
            model_name=model_config.model_name
        )

        if result is not None:
            successful_jailbreaks.append(result)

    # ---------------------------------------------------------
    # Save the Extracted Jailbreak Contexts
    # ---------------------------------------------------------
    # Saving as JSON since we are no longer storing PyTorch tensors
    output_results_path = os.path.join(data_dir, f"jailbreak_contexts_{model_id}.json")

    with open(output_results_path, "w", encoding="utf-8") as f:
        json.dump(successful_jailbreaks, f, indent=4)

    print(f"\nSuccessfully found and extracted {len(successful_jailbreaks)} jailbreaks.")
    print(f"Conversation histories saved to: {output_results_path}")

    print("\n------------------ Job Finished ------------------")