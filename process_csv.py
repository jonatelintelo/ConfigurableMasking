import os
import sys
import pandas as pd
import torch

# Project Modules
import argument_parser as argument_parser
import moe_model_files.model_configurations as model_configurations

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

    initial_safe = {
        "Qwen3-30B-A3B-Instruct-2507": 254,
        "Phi-3.5-MoE-instruct": 310,
        "Mixtral-8x7B-Instruct-v0.1": 256,
        "gpt-oss-20b": 0,
        "Qwen1.5-MoE-A2.7B-Chat": 277,
        "Hunyuan-A13B-Instruct": 0,
        "deepseek-moe-16b-chat": 284,
    }
    
    model_config = model_configurations.models[models[model_id]]
    print(f"\nInitializing: {model_config.model_name}")

    csv_path = os.path.join(root_folder, "results", "graph_numbers", f"{model_config.model_name}.csv")
    print(f"Looking for CSV at: {csv_path}")

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        
        columns_added = False
        
        if 'safe_percentage' not in df.columns:
            df['safe_percentage'] = (df['safe'] / df['total_jailbreaks']) * 100
            df['safe_percentage'] = df['safe_percentage'].round(1)
            columns_added = True
            if print_logging: print("Added 'safe_percentage' column.")
            
        if 'total_prompts' not in df.columns:
            initial_safe_value = initial_safe.get(model_config.model_name, 0)
            df['total_prompts'] = df['total_jailbreaks'] + initial_safe_value
            columns_added = True
            if print_logging: print("Added 'total_prompts' column.")
            
        if 'projected_safe_count' not in df.columns:
            df['projected_safe_count'] = (df['safe_percentage'] / 100) * df['total_prompts']
            df['projected_safe_count'] = df['projected_safe_count'].round(0).astype(int)
            columns_added = True
            if print_logging: print("Added 'projected_safe_count' column.")
            
        if 'final_safe_count' not in df.columns:
            initial_safe_value = initial_safe.get(model_config.model_name, 0)
            df['final_safe_count'] = df['safe'] + initial_safe_value
            df.loc[df['projected_safe_count'] == 0, 'final_safe_count'] = 0
            columns_added = True
            if print_logging: print("Added 'final_safe_count' column.")

        if columns_added:
            df.to_csv(csv_path, index=False)
            if print_logging:
                print(f"File successfully overwritten with new columns at: {csv_path}")
        else:
            if print_logging:
                print("All columns already existed. No changes were made to the CSV.")
                
        if print_logging:
            print("\nPreview of dataframe:")
            print(df.head())
            
    else:
        print(f"Error: CSV file not found at {csv_path}")