import lm_eval
from lm_eval.models.huggingface import HFLM

# ... [Your existing code] ...
# model, tokenizer = model_utils.load_model(models[model_id])

print(f"\nWrapping {model_config.model_name} for lm-eval...")

# Wrap the loaded model and tokenizer
# You can set batch_size="auto" or a specific integer based on your VRAM
eval_model = HFLM(
    pretrained=model,
    tokenizer=tokenizer,
    batch_size="auto" 
)

print("\nStarting MMLU evaluation (5-shot)...")
print("This evaluates all 57 subjects and may take a while.")

# Run the evaluation
results = lm_eval.simple_evaluate(
    model=eval_model,
    tasks=["mmlu"],       # Targets the MMLU task group
    num_fewshot=5,        # 5-shot is the academic standard for MMLU
    log_samples=True      # Logs individual predictions (useful for qualitative analysis)
)

print("\n=== MMLU Results ===")
# Prints a well-formatted markdown table of the results
print(lm_eval.make_table(results))

# Optional: Save the raw results to a JSON file for your paper's artifact submission
import json
output_file = f"mmlu_results_{model_id}.json"
with open(output_file, "w") as f:
    json.dump(results, f, indent=4)
print(f"\nSaved raw evaluation data to {output_file}")