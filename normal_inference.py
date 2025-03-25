import os
import json
import torch
import time
import logging
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import Dataset
from tqdm import tqdm

# ------------------------------------------------------------------------------
# Prompt template (using title, abstract, and MeSH terms)
# ------------------------------------------------------------------------------
PROMPT_DICT = {
    "prompt_input": (
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    )
}

# ------------------------------------------------------------------------------
# Setup logging
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------------------
def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    logger.info("Config loaded from %s", config_path)
    return config

def load_test_data(test_file: str) -> list:
    with open(test_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info("Loaded %d test samples", len(data))
    return data

def format_instruction_input(example: dict) -> dict:
    """
    Build a prompt using the title, abstract, and MeSH terms.
    If the abstract is a list, join it into a single string.
    """
    title = example.get("title", "").strip()
    abstract_val = example.get("abstract", "")
    abstract = (
        " ".join(abstract_val)
        if isinstance(abstract_val, list)
        else abstract_val.strip()
    )
    mesh_terms_list = example.get("mesh_terms", [])
    mesh_terms = ", ".join(mesh_terms_list) if mesh_terms_list else ""
    
    # Adjust the instruction based on MeSH terms.
    if mesh_terms:
        instruction = f"Generate a lay summary of the following scientific article without unnecessary things like 'here is the summary', just give the summary, EMPHASIZING these MeSH terms: {mesh_terms}."
    else:
        instruction = "Generate a lay summary of the following scientific article."

    input_field = f"Title: {title}\nAbstract: {abstract}\n"
    prompt = PROMPT_DICT["prompt_input"].format(
        instruction=instruction, input=input_field
    )
    return {"input_text": prompt}

def extract_generated_summary(full_text: str) -> str:
    """
    Extract the generated summary from the model output.
    If the delimiter '### Response:' is found, return the text after it.
    Otherwise, return the full text.
    """
    if "### Response:" in full_text:
        return full_text.split("### Response:", 1)[1].strip()
    else:
        return full_text.strip()

# ------------------------------------------------------------------------------
# Main function
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Token Length Analysis Script")
    parser.add_argument("--config", type=str, required=True, help="Path to the config JSON file")
    args = parser.parse_args()

    config_path = args.config
    start_time = time.time()
    logger.info("Starting token length analysis script")

    # Load configuration
    config = load_config(config_path)
    model_dir = config["model_dir"]
    test_file = config["test_file"]
    output_file = config["output_file"]
    # New config keys:
    max_input_tokens = config.get("max_input_tokens", None)
    io_output_file = config.get("io_output_file", "input_output_pairs.json")

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device set to: {DEVICE}")
    logger.info(f"Model directory: {model_dir}")
    logger.info(f"Test file: {test_file}")
    logger.info(f"Output file: {output_file}")
    if max_input_tokens:
        logger.info(f"Max input tokens for truncation: {max_input_tokens}")
    else:
        logger.info("No max input tokens specified; no truncation will be applied.")

    # -------------------- Load Model & Tokenizer --------------------
    logger.info("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_dir).to(DEVICE)
    model.eval()
    logger.info("Model and tokenizer loaded.")

    # -------------------- Load and Process Test Data --------------------
    test_data = load_test_data(test_file)
    dataset = Dataset.from_list(test_data)
    dataset = dataset.map(format_instruction_input, batched=False)
    total_samples = len(dataset)
    logger.info("Dataset processing complete. Total samples: %d", total_samples)
    logger.info("Example input (first 500 chars): %s", dataset[0]["input_text"][:500])

    # -------------------- Initialize Generation Pipeline --------------------
    logger.info("Initializing text generation pipeline...")
    summarization_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if DEVICE == "cuda" else -1,
        do_sample=True,
        temperature=config.get("temperature", 0.7),
        top_p=config.get("top_p", 0.9),
        max_new_tokens=config.get("max_new_tokens", 512),
        min_new_tokens=1,
        eos_token_id=tokenizer.eos_token_id,
        batch_size=1,
    )
    logger.info("Pipeline initialized.")

    # -------------------- Token Statistics and IO Pair Analysis --------------------
    input_token_lengths = []
    output_token_lengths = []
    num_truncated_samples = 0
    total_truncated_tokens = 0
    max_truncated_tokens = 0

    io_data = []  # To store input-output pairs along with their stats

    logger.info("Starting generation, token, and truncation analysis...")
    for i, sample in enumerate(tqdm(dataset, desc="Analyzing samples", unit="sample")):
        input_text = sample["input_text"]

        # Compute original token length (without truncation)
        original_tokens = tokenizer.encode(input_text, add_special_tokens=True, truncation=False)
        original_length = len(original_tokens)

        # Apply truncation if max_input_tokens is provided
        if max_input_tokens:
            truncated_tokens = tokenizer.encode(
                input_text, add_special_tokens=True, truncation=True, max_length=max_input_tokens
            )
        else:
            truncated_tokens = original_tokens

        truncated_length = len(truncated_tokens)
        truncation_diff = original_length - truncated_length if original_length > truncated_length else 0

        if truncation_diff > 0:
            num_truncated_samples += 1
            total_truncated_tokens += truncation_diff
            if truncation_diff > max_truncated_tokens:
                max_truncated_tokens = truncation_diff

        # Decode truncated tokens back to string for generation
        truncated_text = tokenizer.decode(truncated_tokens, skip_special_tokens=True)

        # Compute input token lengths (using truncated version as model input)
        input_token_lengths.append(truncated_length)

        try:
            # Generate output using the pipeline with the truncated input
            gen = summarization_pipeline(truncated_text)
            full_generated = (
                gen[0]["generated_text"]
                if gen and isinstance(gen, list) and "generated_text" in gen[0]
                else ""
            )
            # Extract generated summary
            generated_summary = extract_generated_summary(full_generated)
            # Compute output token length
            output_tokens = tokenizer.encode(generated_summary, add_special_tokens=True)
            output_length = len(output_tokens)
            output_token_lengths.append(output_length)
        except Exception as e:
            logger.error("Error processing sample %d: %s", i, e)
            output_token_lengths.append(0)
            generated_summary = ""

        # Record the input-output pair with statistics
        io_data.append({
            "sample_index": i,
            "original_input_text": input_text,
            "truncated_input_text": truncated_text,
            "original_input_token_length": original_length,
            "truncated_input_token_length": truncated_length,
            "tokens_truncated": truncation_diff,
            "generated_summary": generated_summary,
            "output_token_length": output_length
        })

    # Compute overall input and output token statistics
    max_input_length = max(input_token_lengths) if input_token_lengths else 0
    avg_input_length = sum(input_token_lengths) / total_samples if total_samples > 0 else 0
    max_output_length = max(output_token_lengths) if output_token_lengths else 0
    avg_output_length = sum(output_token_lengths) / total_samples if total_samples > 0 else 0

    # Build final analysis results
    analysis_results = {
        "total_samples": total_samples,
        "max_input_token_length": max_input_length,
        "avg_input_token_length": avg_input_length,
        "max_output_token_length": max_output_length,
        "avg_output_token_length": avg_output_length,
        "total_truncated_samples": num_truncated_samples,
        "total_truncated_tokens": total_truncated_tokens,
        "max_truncated_tokens": max_truncated_tokens
    }

    logger.info("Analysis Results:")
    logger.info("Total samples: %d", total_samples)
    logger.info("Max input token length: %d", max_input_length)
    logger.info("Average input token length: %.2f", avg_input_length)
    logger.info("Max output token length: %d", max_output_length)
    logger.info("Average output token length: %.2f", avg_output_length)
    logger.info("Total samples with truncation: %d", num_truncated_samples)
    logger.info("Total truncated tokens across samples: %d", total_truncated_tokens)
    logger.info("Maximum tokens truncated in a single sample: %d", max_truncated_tokens)

    # Save the analysis results to file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(analysis_results, f, indent=4)
    logger.info("Analysis results saved to %s", output_file)

    # Save the input-output pairs for manual review
    with open(io_output_file, "w", encoding="utf-8") as f:
        json.dump(io_data, f, indent=4)
    logger.info("Input-output pairs saved to %s", io_output_file)

    total_time = time.time() - start_time
    logger.info("Script completed in %.2f seconds", total_time)

if __name__ == "__main__":
    main()
