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

    initial_refusals = {
        "Qwen3-30B-A3B-Instruct-2507": 254,
        "Phi-3.5-MoE-instruct": 310,
        "Mixtral-8x7B-Instruct-v0.1": 256,
        "gpt-oss-20b": 0,
        "Qwen1.5-MoE-A2.7B-Chat": 277,
        "Hunyuan-A13B-Instruct": 0,
        "deepseek-moe-16b-chat": 284,
    }
    
    model_config = model_configurations.models[models[model_id]]
    csv_path = "/home/b6aj/jtelintelo.b6aj/ConfigurableMasking/results/graph_numbers/deepseek-moe-16b-chat.csv"
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        
        # Access the 'total' column and convert it to a list
        if 'total' in df.columns:
            total_values = df['total'].tolist()
            if print_logging:
                print(f"Successfully loaded {len(total_values)} values from the 'total' column.")
                print(f"Total values: {total_values}")
        else:
            print("Error: The column 'total' was not found in the CSV.")
    else:
        print(f"Error: CSV file not found at {csv_path}")