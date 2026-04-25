import os
import sys
import pandas as pd
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

# Project Modules
import argument_parser as argument_parser
import moe_model_files.model_configurations as model_configurations


mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Linux Libertine", "Linux Libertine O", "Linux Libertine Display O", "Times New Roman", "serif"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 300,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "lines.linewidth": 1.2,
        "lines.markersize": 3,
        "axes.titleweight": "bold",
        "axes.titlepad": 8.0,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "legend.handlelength": 2.2,
        "legend.columnspacing": 1.2,
        "legend.handletextpad": 0.6,
        "legend.borderaxespad": 0.0,
        "figure.labelsize": 11,
    }
)

# CB_PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
CB_PALETTE = ["#4C78A8", "#E45756", "#F58518", "#54A24B", "#EECA3B", "#72B7B2", "#B279A2", "#FF9DA6"]
LINE_STYLES = ["-", "--", "-.", ":"]
MARKERS = ["o", "s", "^", "D"]

if __name__ == "__main__":
    arguments = argument_parser.parse_arguments()
    root_folder = arguments.root
    print_logging = arguments.print_logging

    font_dir = os.path.join(root_folder, "fonts")
    if os.path.exists(font_dir):
        for font_file in os.listdir(font_dir):
            if font_file.endswith(".ttf"):
                font_manager.fontManager.addfont(os.path.join(font_dir, font_file))

    if print_logging:
        print(f"\nPython version: {sys.version}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA build version: {torch.version.cuda}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        if torch.cuda.is_available():
            print(f"First GPU Name: {torch.cuda.get_device_name(0)}")
            print(f"Test tensor on GPU: {torch.rand(5).cuda().device}")

    all_models = [
        ("Qwen/Qwen3-30B-A3B-Instruct-2507", "Qwen3"),
        ("microsoft/Phi-3.5-MoE-instruct", "Phi"),
        ("mistralai/Mixtral-8x7B-Instruct-v0.1", "Mixtral"),
        ("openai/gpt-oss-20b", "GPT"),
        ("Qwen/Qwen1.5-MoE-A2.7B-Chat", "Qwen1.5"),
        ("tencent/Hunyuan-A13B-Instruct", "Hunyuan"),
        ("deepseek-ai/deepseek-moe-16b-chat", "DeepSeek"),
    ]

    baseline_percentages = [
        ("Qwen/Qwen3-30B-A3B-Instruct-2507", 0.473),
        ("microsoft/Phi-3.5-MoE-instruct", 0.577),
        ("mistralai/Mixtral-8x7B-Instruct-v0.1", 0.477),
        ("openai/gpt-oss-20b", 0.618),
        ("Qwen/Qwen1.5-MoE-A2.7B-Chat", 0.516),
        ("tencent/Hunyuan-A13B-Instruct", 0.488),
        ("deepseek-ai/deepseek-moe-16b-chat", 0.529),
    ]

    main_body_selection = ["Qwen3", "Mixtral", "DeepSeek"]

    all_data = []
    print("\nLoading data for all models...")

    for model_path, short_name in all_models:
        if model_path not in model_configurations.models:
            print(f"Skipping {model_path} - not in model_configurations")
            continue

        model_config = model_configurations.models[model_path]
        model_name = model_config.model_name

        csv_path = os.path.join(root_folder, "results", "graph_numbers", "jailbreak", f"{model_name}.csv")

        if os.path.exists(csv_path):
            df_temp = pd.read_csv(csv_path, dtype={"lambda": str})
            df_temp["model_name"] = short_name
            df_temp["safety_rate"] = df_temp["final_safe_count"] / df_temp["total_prompts"]

            df_temp["in_main_body"] = short_name in main_body_selection

            all_data.append(df_temp)
            print(f"Successfully loaded: {short_name}")
        else:
            print(f"WARNING: Missing CSV for {model_name} at {csv_path}")

    if not all_data:
        print("\nError: No CSV files were found. Exiting.")
        sys.exit(1)

    df = pd.concat(all_data, ignore_index=True)

    def create_ablation_figure(data_subset, rows, cols, figsize, filename_suffix, legend_y=1.18, xlabel_y=0.08):
        fig, axs = plt.subplots(rows, cols, figsize=figsize, sharex=True, sharey=True)

        if rows * cols > 1:
            axs = axs.ravel()
        else:
            axs = [axs]

        models_in_subset = data_subset["model_name"].unique()
        unique_lambdas = data_subset["lambda"].unique()

        # Create a mapping for quick baseline lookup
        # This maps the short_name to the baseline percentage
        model_to_baseline = {short: val for (_, short), (_, val) in zip(all_models, baseline_percentages)}

        lines = []
        labels = []

        for i, model in enumerate(models_in_subset):
            ax = axs[i]
            model_df = data_subset[data_subset["model_name"] == model]

            # --- ADDED: Baseline Horizontal Line ---
            if model in model_to_baseline:
                ax.axhline(y=model_to_baseline[model], color="black", linestyle="-.", linewidth=0.8, alpha=0.6, zorder=1, label="Baseline")
            # ---------------------------------------

            for l_idx, lam in enumerate(unique_lambdas):
                lam_df = model_df[model_df["lambda"] == lam].sort_values(by="alpha")

                (line,) = ax.plot(
                    lam_df["alpha"],
                    lam_df["safety_rate"],
                    color=CB_PALETTE[l_idx % len(CB_PALETTE)],
                    linestyle=LINE_STYLES[l_idx % len(LINE_STYLES)],
                    marker=MARKERS[l_idx % len(MARKERS)],
                    label=lam,
                    zorder=2,  # Ensure data is above the baseline
                )

                if i == 0:
                    lines.append(line)
                    labels.append(lam)

            ax.set_title(model)
            ax.set_ylim(-0.05, 1.05)

        fig.supylabel("Success Rate")
        fig.supxlabel(r"$\alpha$", y=xlabel_y)

        fig.tight_layout()

        # Legend
        handles, labels = ax.get_legend_handles_labels()
        # Filter handles to ensure 'Baseline' and Lambda values are handled correctly in the legend if needed
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, legend_y),
            ncol=len(unique_lambdas) + 1,  # +1 for the baseline label
            title=r"$\lambda$ Penalty Weight:",
        )

        pdf_path = os.path.join(root_folder, "results", f"{filename_suffix}.pdf")
        fig.savefig(pdf_path, bbox_inches="tight")
        print(f"Saved to {pdf_path}")

    MODE = "jailbreak"

    print("\nGenerating ACM CCS Main Body Plot...")
    df_main = df[df["in_main_body"] == True]
    create_ablation_figure(
        df_main,
        rows=1,
        cols=3,
        figsize=(7, 2.2),
        filename_suffix=f"{MODE}_ablation_main",
        legend_y=1.18,  # Default main text spacing
        xlabel_y=0.08,  # Default main text spacing
    )

    print("\nGenerating ACM CCS Appendix Plot...")
    df_appendix = df[df["in_main_body"] == False]
    if not df_appendix.empty:
        create_ablation_figure(
            df_appendix,
            rows=2,
            cols=2,
            figsize=(7, 6.6),
            filename_suffix=f"{MODE}_ablation_appendix",
            legend_y=1.09,  # Adjusted for 2x height (maintains 0.39 inches absolute)
            xlabel_y=0.04,  # Adjusted for 2x height (maintains 0.17 inches absolute)
        )
    else:
        print("No models left for the appendix plot.")

    print("\n------------------ Job Finished ------------------\n")
