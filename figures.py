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

    # All models evaluated
    all_models = [
        "Qwen/Qwen3-30B-A3B-Instruct-2507",  # 0
        "microsoft/Phi-3.5-MoE-instruct",  # 1
        "mistralai/Mixtral-8x7B-Instruct-v0.1",  # 2
        "openai/gpt-oss-20b",  # 3 (Added back for the appendix)
        "Qwen/Qwen1.5-MoE-A2.7B-Chat",  # 4
        "tencent/Hunyuan-A13B-Instruct",  # 5 (Added back for the appendix)
        "deepseek-ai/deepseek-moe-16b-chat",  # 6
    ]

    # Models selected for the MAIN BODY of the paper (1x3 grid)
    main_body_selection = [
        "Qwen/Qwen3-30B-A3B-Instruct-2507",      
        "mistralai/Mixtral-8x7B-Instruct-v0.1",  
        "deepseek-ai/deepseek-moe-16b-chat"      
    ]

    # 2. Iterate through all models and combine data
    all_data = []
    print("\nLoading data for all models...")
    
    for model_path in all_models:
        # Check if the model config exists, skip gracefully if not loaded in your env
        if model_path not in model_configurations.models:
            print(f"Skipping {model_path} - not in model_configurations")
            continue
            
        model_config = model_configurations.models[model_path]
        model_name = model_config.model_name
        
        csv_path = os.path.join(root_folder, "results", "graph_numbers", f"{model_name}.csv")
        
        if os.path.exists(csv_path):
            df_temp = pd.read_csv(csv_path)
            df_temp['model_name'] = model_name
            df_temp['safety_rate'] = df_temp['final_safe_count'] / df_temp['total_prompts']
            
            # Keep a boolean flag for easy splitting
            df_temp['in_main_body'] = model_path in main_body_selection
            
            all_data.append(df_temp)
            print(f"Successfully loaded: {model_name}")
        else:
            print(f"WARNING: Missing CSV for {model_name} at {csv_path}")

    if not all_data:
        print("\nError: No CSV files were found. Exiting.")
        sys.exit(1)

    df = pd.concat(all_data, ignore_index=True)

    # 3. Plotting Configuration (ACM CCS Standards)
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'font.size': 9,
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 9,
        'legend.title_fontsize': 9,
    })

    # --- MAIN BODY PLOT (1x3 Grid) ---
    print("\nGenerating ACM CCS Main Body Plot...")
    df_main = df[df['in_main_body'] == True]

    g_main = sns.relplot(
        data=df_main,
        x="alpha", 
        y="safety_rate", 
        hue="lambda",      
        style="lambda",    
        col="model_name",  
        col_wrap=3,        
        kind="line", 
        markers=True, 
        dashes=True,
        palette="colorblind",
        height=2.0,        
        aspect=1.05,
        facet_kws={'sharex': True, 'sharey': True} # Explicitly share axes
    )

    g_main.set_axis_labels(r"$\alpha$ Threshold", "Safety Rate")
    g_main.set_titles(col_template="{col_name}")

    # Move legend to a horizontal row above the subplots
    sns.move_legend(
        g_main, "lower center", 
        bbox_to_anchor=(0.5, 1.05), 
        ncol=4, # 4 items in the legend, so they form a single horizontal line
        title=r"$\lambda$ Penalty Weight:", 
        frameon=False
    )

    plt.tight_layout()
    output_main = os.path.join(root_folder, "results", "ablation_lines_main_body_ccs.pdf")
    plt.savefig(output_main, dpi=300, bbox_inches='tight')
    print(f"Main body plot saved: {output_main}")


    # --- APPENDIX PLOT (2x2 Grid) ---
    print("\nGenerating ACM CCS Appendix Plot...")
    df_appendix = df[df['in_main_body'] == False]

    if not df_appendix.empty:
        # We use a 2x2 grid for the remaining 4 models to keep aspect ratios consistent
        g_app = sns.relplot(
            data=df_appendix,
            x="alpha", 
            y="safety_rate", 
            hue="lambda",      
            style="lambda",    
            col="model_name",  
            col_wrap=2,        # 2 columns for a 2x2 layout      
            kind="line", 
            markers=True, 
            dashes=True,
            palette="colorblind",
            height=2.0,        # Kept identical to main body for visual consistency
            aspect=1.05,       
            facet_kws={'sharex': True, 'sharey': True}
        )

        g_app.set_axis_labels(r"$\alpha$ Threshold", "Safety Rate")
        g_app.set_titles(col_template="{col_name}")

        # Move legend to a horizontal row above the subplots
        sns.move_legend(
            g_app, "lower center", 
            bbox_to_anchor=(0.5, 1.05), 
            ncol=4, 
            title=r"$\lambda$ Penalty Weight:", 
            frameon=False
        )

        plt.tight_layout()
        output_app = os.path.join(root_folder, "results", "ablation_lines_appendix_ccs.pdf")
        plt.savefig(output_app, dpi=300, bbox_inches='tight')
        print(f"Appendix plot saved: {output_app}")
    else:
        print("No models left for the appendix plot.")