import os
import json
import torch
import time
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import Dataset
from rouge_score import rouge_scorer
from evaluate import load
from tqdm import tqdm

# ------------------------------------------------------------------------------
# Prompt template (using only title, abstract, and MeSH terms)
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
    Build a prompt using only the title, abstract, and MeSH terms.
    If the abstract is a list, join it into a single string.
    """
    title = example.get("title", "").strip()
    abstract_val = example.get("abstract", "")
    abstract = (
        " ".join(abstract_val)
        if isinstance(abstract_val, list)
        else abstract_val.strip()
    )
    #mesh_terms_list = example.get("mesh_terms", [])
    #mesh_terms = ", ".join(mesh_terms_list)

    # Adjust instruction based on MeSH terms.
    """
    if mesh_terms:
        instruction = f"Generate a lay summary of the following scientific article, emphasizing these MeSH terms: {mesh_terms}."
    else:
        instruction = "Generate a lay summary of the following scientific article."
    """
    instruction = "Generate a lay summary of the following scientific article."
    # Reorder the input field to highlight MeSH terms.
    # input_field = f"Title: {title}\nMeSH Terms: {mesh_terms}\nAbstract: {abstract}\n"
    input_field = f"Title: {title}\nAbstract: {abstract}\n"
    prompt = PROMPT_DICT["prompt_input"].format(
        instruction=instruction, input=input_field
    )
    return {"input_text": prompt}


# ------------------------------------------------------------------------------
# Main function
# ------------------------------------------------------------------------------
def main():
    config_path = "inference_config.json"
    start_time = time.time()
    logger.info("Starting instruction summarization script")

    # Load configuration from file
    config = load_config(config_path)
    model_dir = config["model_dir"]
    test_file = config["test_file"]
    output_file = config["output_file"]
    ACCESS_TOKEN = config.get("access_token", None)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device set to: {DEVICE}")
    logger.info(f"Model dir: {model_dir}")
    logger.info(f"Test file: {test_file}")
    logger.info(f"Output file: {output_file}")

    # -------------------- Model & Tokenizer --------------------
    logger.info("Loading finetuned model and tokenizer...")
    model_load_start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_auth_token=ACCESS_TOKEN)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, use_auth_token=ACCESS_TOKEN
    ).to(DEVICE)
    model_load_end = time.time()
    logger.info(f"Model loaded in {model_load_end - model_load_start:.2f} seconds")

    # -------------------- Load and Process Test Data --------------------
    test_data = load_test_data(test_file)
    dataset = Dataset.from_list(test_data)
    dataset = dataset.map(format_instruction_input, batched=False)
    logger.info("Dataset processing complete. Total samples: %d", len(dataset))
    logger.info(
        "First input example (first 500 chars): %s", dataset[0]["input_text"][:500]
    )

    # -------------------- Pipeline Initialization --------------------
    logger.info("Initializing text generation pipeline...")
    pipeline_start = time.time()
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
    pipeline_end = time.time()
    logger.info(f"Pipeline initialized in {pipeline_end - pipeline_start:.2f} seconds")

    # -------------------- Generate Summaries --------------------
    logger.info("Starting summary generation...")
    total_samples = len(dataset["input_text"])
    generated_texts = []
    overall_start = time.time()

    for i, input_text in enumerate(
        tqdm(dataset["input_text"], desc="Generating summaries", unit="sample")
    ):
        sample_start = time.time()
        try:
            gen = summarization_pipeline(input_text)
            full_generated = (
                gen[0]["generated_text"]
                if gen and isinstance(gen, list) and "generated_text" in gen[0]
                else ""
            )
            # Extract model's answer by splitting on the prompt delimiter
            if "### Response:" in full_generated:
                generated_summary = full_generated.split("### Response:", 1)[1].strip()
            else:
                generated_summary = full_generated.strip()
            generated_texts.append(generated_summary)
        except Exception as e:
            logger.error("Error processing sample %d: %s", i, e)
            generated_texts.append("")
        sample_time = time.time() - sample_start
        avg_time = (time.time() - overall_start) / (i + 1)
        remaining_time = avg_time * (total_samples - i - 1)
        logger.info(
            f"Sample {i+1}/{total_samples} processed in {sample_time:.2f} sec; remaining ~{remaining_time:.2f} sec"
        )

    total_generation_time = time.time() - overall_start
    logger.info(f"Summarization completed in {total_generation_time:.2f} seconds")

    # -------------------- Save Summaries --------------------
    summaries = []
    for i, gen_text in enumerate(generated_texts):
        if gen_text:
            summaries.append({"id": test_data[i]["id"], "generated_summary": gen_text})
        else:
            logger.warning("No output for sample %d", i)
    logger.info("Generated summaries for %d samples", len(summaries))
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=4)
    logger.info("Summaries saved to %s", output_file)

    # -------------------- ROUGE Evaluation --------------------
    logger.info("Starting ROUGE evaluation...")
    # For evaluation, use the 'summary' field from the test data.
    # If summary is a list, join it into a single string.
    gold_data = {}
    for item in test_data:
        summ = item.get("summary", "")
        if isinstance(summ, list):
            summ = " ".join(summ)
        gold_data[item["id"]] = summ
    generated_data = {item["id"]: item["generated_summary"] for item in summaries}
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True, tokenizer=tokenizer
    )
    rouge_scores = []
    for id_, gold_summary in gold_data.items():
        if id_ in generated_data:
            score = scorer.score(gold_summary, generated_data[id_])
            rouge_scores.append(score)
    if rouge_scores:
        avg_rouge1 = sum(s["rouge1"].fmeasure for s in rouge_scores) / len(rouge_scores)
        avg_rouge2 = sum(s["rouge2"].fmeasure for s in rouge_scores) / len(rouge_scores)
        avg_rougeL = sum(s["rougeL"].fmeasure for s in rouge_scores) / len(rouge_scores)
        logger.info(
            "ROUGE results: rouge-1: %.4f, rouge-2: %.4f, rouge-L: %.4f",
            avg_rouge1,
            avg_rouge2,
            avg_rougeL,
        )
    else:
        logger.warning("No ROUGE scores computed; check data")

    # -------------------- SARI Evaluation --------------------
    logger.info("Starting SARI evaluation...")
    sari = load("sari")
    sources = [dataset[i]["input_text"] for i in range(total_samples)]
    predictions = list(generated_data.values())
    references = [[gold_data[id_]] for id_ in generated_data.keys()]
    sari_result = sari.compute(
        sources=sources, predictions=predictions, references=references
    )
    logger.info("SARI result: %.4f", sari_result.get("sari", 0.0))

    total_script_time = time.time() - start_time
    logger.info("Script completed in %.2f seconds", total_script_time)


if __name__ == "__main__":
    main()
