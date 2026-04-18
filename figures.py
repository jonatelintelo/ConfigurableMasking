import os
import sys
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# Project Modules
import argument_parser as argument_parser
import moe_model_files.model_configurations as model_configurations

if __name__ == "__main__":
    arguments = argument_parser.parse_arguments()
    root_folder = arguments.root
    print_logging = arguments.print_logging

    # 1. System Logging
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
        # "openai/gpt-oss-20b",  # 3
        "Qwen/Qwen1.5-MoE-A2.7B-Chat",  # 4
        # "tencent/Hunyuan-A13B-Instruct",  # 5
        "deepseek-ai/deepseek-moe-16b-chat",  # 6
    ]

    # 2. Iterate through all models and combine data
    all_data = []
    print("\nLoading data for all models...")
    
    for model_path in models:
        model_config = model_configurations.models[model_path]
        model_name = model_config.model_name
        
        csv_path = os.path.join(root_folder, "results", "graph_numbers", f"{model_name}.csv")
        
        if os.path.exists(csv_path):
            df_temp = pd.read_csv(csv_path)
            # Add a column for the model name so Seaborn knows how to split the subplots
            df_temp['model_name'] = model_name
            # Calculate the metric needed for the Y-axis
            df_temp['safety_rate'] = df_temp['final_safe_count'] / df_temp['total_prompts']
            
            all_data.append(df_temp)
            print(f"Successfully loaded: {model_name}")
        else:
            print(f"WARNING: Missing CSV for {model_name} at {csv_path}")

    # Safety check in case paths are wrong
    if not all_data:
        print("\nError: No CSV files were found. Exiting.")
        sys.exit(1)

    # Combine the list of dataframes into one master dataframe
    df = pd.concat(all_data, ignore_index=True)

    # 3. Plotting Configuration (ACM CCS Standards)
    print("\nGenerating ACM CCS formatted plot...")
    sns.set_theme(style="whitegrid")

    plt.rcParams.update({
        'font.size': 9,
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'legend.title_fontsize': 9,
    })

    # 4. Generate the Faceted Plot
    g = sns.relplot(
        data=df,
        x="alpha", 
        y="safety_rate", 
        hue="lambda",      
        style="lambda",    
        col="model_name",  
        col_wrap=4,        
        kind="line", 
        markers=True, 
        dashes=True,
        palette="colorblind",
        height=1.8,        
        aspect=0.95        
    )

    # Customize the labels and axes
    g.set_axis_labels(r"$\alpha$ Threshold", "Safety Rate")
    g.set_titles(col_template="{col_name}")

    # Move legend to the empty 8th subplot space
    sns.move_legend(g, "center", bbox_to_anchor=(0.85, 0.25), frameon=True)

    # Tighten layout and export
    plt.tight_layout()
    output_filename = os.path.join(root_folder, "results", "ablation_lines_ccs.pdf")
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved successfully to: {output_filename}")
    
    # Optional: Keep plt.show() if you are running this interactively to preview
    # plt.show()