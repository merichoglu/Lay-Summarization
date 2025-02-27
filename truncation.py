import json
import os
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

# Load tokenizer
model_name = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# explicitly set pad_token to eos_token if not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Model's max context length
context_length = 128000  # 128k tokens

output_dir = "token_stats"
os.makedirs(output_dir, exist_ok=True)

datasets = {
    "train": "prepared_data/train_prepared.json",
    "val": "prepared_data/val_prepared.json",
    "test": "prepared_data/test_prepared.json",
}


def process_dataset(dataset_name, file_path):
    """Processes a dataset file to compute token length statistics."""

    # Read entire JSON file as a list
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            dataset = json.load(f)
        except json.JSONDecodeError as e:
            print(f"❌ Error: Could not load {dataset_name} due to malformed JSON: {e}")
            return

    token_lengths = []
    prompt_lengths = []
    num_truncated = 0
    total_samples = len(dataset)

    for data in dataset:
        # Retrieve and format components
        title = data.get("title", "").strip()
        abstract_val = data.get("abstract", "")
        abstract = (
            " ".join(abstract_val)
            if isinstance(abstract_val, list)
            else abstract_val.strip()
        )
        mesh_terms_list = data.get("mesh_terms", [])
        mesh_terms = (
            ", ".join(mesh_terms_list) if mesh_terms_list != ["NO_MESH_TERMS"] else ""
        )

        # Adjust instruction based on MeSH terms
        instruction = (
            f"Generate a lay summary of the following scientific article, emphasizing these MeSH terms: {mesh_terms}."
            if mesh_terms
            else "Generate a lay summary of the following scientific article."
        )

        # Format input field
        input_field = (
            f"Title: {title}\nMeSH Terms: {mesh_terms}\nAbstract: {abstract}\n"
        )

        summary = data.get("summary", [])
        output_text = " ".join(summary) if isinstance(summary, list) else summary

        prompt = f"{instruction}\n\n{input_field}"
        full_text = prompt + output_text

        if not full_text.endswith(tokenizer.eos_token):
            full_text += tokenizer.eos_token

        tokenized_full = tokenizer(
            full_text,
            max_length=context_length,
            padding="max_length",
            truncation=True,
        )
        token_length = sum(tokenized_full["attention_mask"])  # Count non-padding tokens

        # tokenize prompt to get its length
        tokenized_prompt = tokenizer(
            prompt,
            max_length=context_length,
            padding="max_length",
            truncation=True,
        )
        prompt_len = sum(tokenized_prompt["attention_mask"])  # Count non-padding tokens

        token_lengths.append(token_length)
        prompt_lengths.append(prompt_len)

        if token_length >= context_length:
            num_truncated += 1

    # Compute statistics
    truncation_ratio = (num_truncated / total_samples) * 100 if total_samples > 0 else 0
    max_length = max(token_lengths) if token_lengths else 0
    avg_length = np.mean(token_lengths) if token_lengths else 0
    avg_prompt_length = np.mean(prompt_lengths) if prompt_lengths else 0

    stats = {
        "total_samples": total_samples,
        "truncated_samples": num_truncated,
        "truncation_ratio (%)": round(truncation_ratio, 2),
        "max_token_length": max_length,
        "average_token_length": round(avg_length, 2),
        "average_prompt_length": round(avg_prompt_length, 2),
    }

    # Save statistics
    stats_file = os.path.join(output_dir, f"{dataset_name}_stats.json")
    with open(stats_file, "w", encoding="utf-8") as f_out:
        json.dump(stats, f_out, indent=4)

    print(f"✅ Processed {dataset_name}: {total_samples} samples")
    print(f"📊 Truncated Samples: {num_truncated} ({truncation_ratio:.2f}%)")
    print(
        f"📏 Max Length: {max_length}, Avg Length: {avg_length:.2f}, Avg Prompt Length: {avg_prompt_length:.2f}"
    )
    print(f"📁 Stats saved to: {stats_file}")

    # Plot histogram of token lengths
    plt.figure(figsize=(10, 5))
    plt.hist(token_lengths, bins=50, alpha=0.75, label="Full Input")
    plt.hist(prompt_lengths, bins=50, alpha=0.5, label="Prompt Only")
    plt.axvline(
        context_length,
        color="r",
        linestyle="dashed",
        linewidth=2,
        label="Context Length (128k)",
    )
    plt.xlabel("Token Length")
    plt.ylabel("Number of Inputs")
    plt.title(f"Token Length Distribution - {dataset_name}")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_histogram.png"))
    plt.close()


# Process each dataset
for dataset_name, file_path in datasets.items():
    if os.path.exists(file_path):
        process_dataset(dataset_name, file_path)
    else:
        print(f"⚠️ File not found: {file_path}")

print("🎯 All datasets processed. Results stored in 'token_stats/' folder.")
