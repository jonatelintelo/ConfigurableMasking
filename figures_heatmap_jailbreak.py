import os
import torch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import seaborn as sns

import argument_parser as argument_parser

# --- ACM CCS 2026 Styling Parameters ---
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
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "legend.frameon": False,
        "figure.labelsize": 11,
    }
)


def calculate_expert_frequencies(topk_data, num_experts=None):
    """
    Calculates the activation frequency of each expert per layer.
    """
    num_layers = topk_data[0].shape[1]

    if num_experts is None:
        max_expert_idx = max(int(tensor.max().item()) for tensor in topk_data)
        num_experts = max_expert_idx + 1

    expert_counts = np.zeros((num_layers, num_experts))

    for tensor in topk_data:
        for layer_idx in range(num_layers):
            layer_experts = tensor[:, layer_idx, :].flatten().cpu().numpy()
            unique, counts = np.unique(layer_experts, return_counts=True)
            expert_counts[layer_idx, unique.astype(int)] += counts

    row_sums = expert_counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    frequencies = expert_counts / row_sums

    return frequencies, num_experts


def plot_experiment_comparison(models_to_plot, exp1_dict, exp2_dict, exp1_name, exp2_name, save_path, title_prefix=""):
    """
    Plots a grid comparing two experiments for the same models.
    Columns = Experiments, Rows = Models. Optimized for ACM CCS.
    """
    num_models = len(models_to_plot)
    if num_models == 0:
        return

    cols = 2
    rows = num_models

    # TIGHT LAYOUT FIX: layout="constrained" and reduced height multiplier
    fig, axes = plt.subplots(rows, cols, figsize=(7.0, 2.0 * rows), layout="constrained")
    if rows == 1:
        axes = np.array([axes])

    max_abs_diff = 0
    for m in models_to_plot:
        if m in exp1_dict:
            max_abs_diff = max(max_abs_diff, np.max(np.abs(exp1_dict[m])))
        if m in exp2_dict:
            max_abs_diff = max(max_abs_diff, np.max(np.abs(exp2_dict[m])))

    for row_idx, model_name in enumerate(models_to_plot):
        data_exp1 = exp1_dict.get(model_name)
        data_exp2 = exp2_dict.get(model_name)

        # Plot Experiment 1 (Left Column)
        if data_exp1 is not None:
            ax1 = axes[row_idx, 0]
            # Ticks removed via xticklabels=False, yticklabels=False
            sns.heatmap(
                data_exp1, cmap="RdBu_r", center=0, vmin=-max_abs_diff, vmax=max_abs_diff, square=True, ax=ax1, cbar=False, xticklabels=False, yticklabels=False
            )
            ax1.tick_params(left=False, bottom=False)
            ax1.set_title(f"{model_name}\n({exp1_name})", pad=8)
            ax1.invert_yaxis()
            ax1.set_ylabel("Layer Index")
            if row_idx == rows - 1:
                ax1.set_xlabel("Expert Index")

        # Plot Experiment 2 (Right Column)
        if data_exp2 is not None:
            ax2 = axes[row_idx, 1]
            # Ticks removed via xticklabels=False, yticklabels=False
            sns.heatmap(
                data_exp2, cmap="RdBu_r", center=0, vmin=-max_abs_diff, vmax=max_abs_diff, square=True, ax=ax2, cbar=False, xticklabels=False, yticklabels=False
            )
            ax2.tick_params(left=False, bottom=False)
            ax2.set_title(f"{model_name}\n({exp2_name})", pad=8)
            ax2.invert_yaxis()
            if row_idx == rows - 1:
                ax2.set_xlabel("Expert Index")

    # TIGHT LAYOUT FIX: Adjusted colorbar parameters to work natively with constrained layout
    mappable = axes[0, 0].collections[0] if exp1_dict.get(models_to_plot[0]) is not None else axes[0, 1].collections[0]
    cbar = fig.colorbar(mappable, ax=axes.ravel().tolist(), orientation="vertical", shrink=0.8, aspect=30)
    cbar.set_label("Change in Frequency (Steered - Baseline)", rotation=270, labelpad=15)

    if title_prefix:
        fig.suptitle(f"{title_prefix}", y=1.02)

    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    print(f"Plot saved to: {save_path}")


def plot_single_experiment(models_to_plot, diff_data_dict, exp_name, save_path, title_prefix=""):
    """
    Plots a grid of heatmaps for models that only have one experiment.
    """
    num_models = len(models_to_plot)
    if num_models == 0:
        return

    cols = min(num_models, 3)
    rows = int(np.ceil(num_models / cols))

    # TIGHT LAYOUT FIX: layout="constrained" and reduced height multiplier
    fig, axes = plt.subplots(rows, cols, figsize=(7.0, 2.0 * rows), layout="constrained")
    axes_flat = np.array([axes]) if num_models == 1 else axes.flatten()

    max_abs_diff = max(np.max(np.abs(diff_data_dict[m])) for m in models_to_plot)

    for idx, model_name in enumerate(models_to_plot):
        ax = axes_flat[idx]
        diff_freq = diff_data_dict[model_name]

        # Ticks removed via xticklabels=False, yticklabels=False
        sns.heatmap(
            diff_freq, cmap="RdBu_r", center=0, vmin=-max_abs_diff, vmax=max_abs_diff, square=True, ax=ax, cbar=False, xticklabels=False, yticklabels=False
        )
        ax.tick_params(left=False, bottom=False)
        ax.set_title(f"{model_name}\n({exp_name})", pad=8)
        ax.invert_yaxis()

    for idx in range(num_models, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    for idx in range(num_models):
        ax = axes_flat[idx]
        if idx % cols == 0:
            ax.set_ylabel("Layer Index")
        if (idx + cols) >= len(axes_flat) or (len(axes_flat) > num_models and (idx + cols) >= num_models):
            ax.set_xlabel("Expert Index")

    # TIGHT LAYOUT FIX: Adjusted colorbar parameters to work natively with constrained layout
    mappable = axes_flat[0].collections[0]
    cbar = fig.colorbar(mappable, ax=axes_flat.tolist(), orientation="vertical", shrink=0.8, aspect=30)
    cbar.set_label("Change in Frequency (Steered - Baseline)", rotation=270, labelpad=15)

    if title_prefix:
        fig.suptitle(f"{title_prefix}", y=1.02)

    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    print(f"Plot saved to: {save_path}")


def main():
    arguments = argument_parser.parse_arguments()
    root_folder = arguments.root

    font_dir = os.path.join(root_folder, "fonts")
    if os.path.exists(font_dir):
        for font_file in os.listdir(font_dir):
            if font_file.endswith(".ttf"):
                font_manager.fontManager.addfont(os.path.join(font_dir, font_file))

    all_models = [
        ("Qwen/Qwen3-30B-A3B-Instruct-2507", "Qwen3"),
        ("microsoft/Phi-3.5-MoE-instruct", "Phi"),
        ("mistralai/Mixtral-8x7B-Instruct-v0.1", "Mixtral"),
        ("openai/gpt-oss-20b", "GPT"),
        ("Qwen/Qwen1.5-MoE-A2.7B-Chat", "Qwen1.5"),
        ("tencent/Hunyuan-A13B-Instruct", "Hunyuan"),
        ("deepseek-ai/deepseek-moe-16b-chat", "DeepSeek"),
    ]

    experiments = ["jailbreak", "adult_refusal"]
    data_dicts = {exp: {} for exp in experiments}

    print("Loading tensors and calculating frequencies...")
    for repo_path, short_name in all_models:
        model_filename = repo_path.split("/")[-1]

        for exp in experiments:
            baseline_path = os.path.join(root_folder, "results", "topk", exp, f"{model_filename}_baseline_topk.pt")
            steered_path = os.path.join(root_folder, "results", "topk", exp, f"{model_filename}_steered_topk.pt")

            if os.path.exists(baseline_path) and os.path.exists(steered_path):
                baseline_data = torch.load(baseline_path)
                steered_data = torch.load(steered_path)

                baseline_freq, num_experts = calculate_expert_frequencies(baseline_data)
                steered_freq, _ = calculate_expert_frequencies(steered_data, num_experts=num_experts)

                # Store data using the short_name
                data_dicts[exp][short_name] = steered_freq - baseline_freq

    # Identify models using the new short names
    intersection_models = [m for m in data_dicts["jailbreak"].keys() if m in data_dicts["adult_refusal"]]
    single_exp_models = [m for m in data_dicts["jailbreak"].keys() if m not in data_dicts["adult_refusal"]]

    # Pick the 2 most important models for the main paper 2x2 grid (Updated to short names)
    main_intersection = [m for m in ["GPT", "Hunyuan"] if m in intersection_models]

    appendix_intersection = [m for m in intersection_models if m not in main_intersection]
    appendix_single = single_exp_models

    save_dir = os.path.join(root_folder, "results", "figures")
    os.makedirs(save_dir, exist_ok=True)

    if main_intersection:
        print(f"\nGenerating Main 2x2 Figure with: {main_intersection}")
        plot_experiment_comparison(
            models_to_plot=main_intersection,
            exp1_dict=data_dicts["jailbreak"],
            exp2_dict=data_dicts["adult_refusal"],
            exp1_name="Multi-turn Jailbreak Defense",
            exp2_name="Adult-Content Generation",
            save_path=os.path.join(save_dir, "main_comparison_heatmaps.pdf"),
        )

    if appendix_intersection:
        print(f"\nGenerating Appendix 2x2 Figure with: {appendix_intersection}")
        plot_experiment_comparison(
            models_to_plot=appendix_intersection,
            exp1_dict=data_dicts["jailbreak"],
            exp2_dict=data_dicts["adult_refusal"],
            exp1_name="Multi-turn Jailbreak Defense",
            exp2_name="Adult-Content Generation",
            save_path=os.path.join(save_dir, "appendix_comparison_heatmaps.pdf"),
        )

    if appendix_single:
        print(f"\nGenerating Appendix Single-Experiment Figure with: {appendix_single}")
        plot_single_experiment(
            models_to_plot=appendix_single,
            diff_data_dict=data_dicts["jailbreak"],
            exp_name="Adult-Content Generation",
            save_path=os.path.join(save_dir, "appendix_single_heatmaps.pdf"),
        )


if __name__ == "__main__":
    main()
