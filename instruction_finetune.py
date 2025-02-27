import os
import json
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    set_seed,
)

# Use SFTTrainer from the trl package for supervised finetuning.
from trl import SFTTrainer
from datasets import Dataset, DatasetDict


def load_config(config_path):
    """Load configuration parameters from a JSON file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_data(data_dir, train_fraction=1.0):
    """
    Load datasets from the specified directory.
    Expected files: train_prepared.json, val_prepared.json, test_prepared.json.
    Optionally truncates the training set to a fraction.
    (Unnecessary file existence checks have been removed, assuming well maintained data.)
    """
    datasets = {}
    for split in ["train", "val", "test"]:
        file_path = os.path.join(data_dir, f"{split}_prepared.json")
        print(f"Loading {split} data from {file_path}...")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"{split} dataset contains {len(data)} records.")
        if split == "train" and train_fraction < 1.0:
            num_samples = int(len(data) * train_fraction)
            data = data[:num_samples]
            print(f"Using {num_samples} samples from the training dataset.")
        datasets[split] = Dataset.from_list(data)
    return DatasetDict(datasets)


# ------------------------------------------------------------------------------
# Define prompt templates for instruction finetuning.
# ------------------------------------------------------------------------------
PROMPT_DICT = {
    "prompt_input": "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
    "prompt_no_input": "### Instruction:\n{instruction}\n\n### Response:\n",
}


def preprocess_instruction(example, tokenizer, max_seq_length):
    """
    Preprocess an example for instruction finetuning by appending instruction, title,
    abstract, MeSH terms and summary as the expected output.
    """
    # Retrieve and format components.
    title = example.get("title", "").strip()
    abstract_val = example.get("abstract", "")
    abstract = " ".join(abstract_val) if isinstance(abstract_val, list) else abstract_val.strip()
    mesh_terms_list = example.get("mesh_terms", [])
    mesh_terms = ", ".join(mesh_terms_list)
    
    # Adjust instruction based on MeSH terms.
    if mesh_terms:
        instruction = f"Generate a lay summary of the following scientific article, emphasizing these MeSH terms: {mesh_terms}."
    else:
        instruction = "Generate a lay summary of the following scientific article."
    
    # Reorder the input field to highlight MeSH terms.
    input_field = f"Title: {title}\nMeSH Terms: {mesh_terms}\nAbstract: {abstract}\n"
    
    # Use summary as the output text.
    summary = example.get("summary", [])
    output_text = " ".join(summary) if isinstance(summary, list) else summary

    # Create the prompt using a predefined template.
    prompt = PROMPT_DICT["prompt_input"].format(
        instruction=instruction, input=input_field
    )
    
    full_text = prompt + output_text

    # For LLaMA-based instruct models, ensure the full text ends with the EOS token.
    if not full_text.endswith(tokenizer.eos_token):
        full_text += tokenizer.eos_token

    # Tokenize the full text.
    tokenized_full = tokenizer(
        full_text,
        max_length=max_seq_length,
        padding="max_length",
        truncation=True,
    )
    
    # Tokenize the prompt separately to determine its length.
    tokenized_prompt = tokenizer(
        prompt,
        max_length=max_seq_length,
        padding="max_length",
        truncation=True,
    )
    
    input_ids = tokenized_full["input_ids"]
    attention_mask = tokenized_full["attention_mask"]
    
    # Determine the prompt length (count of non-padding tokens).
    prompt_len = sum(tokenized_prompt["attention_mask"])
    labels = input_ids.copy()
    
    # Mask out the prompt tokens so that the loss is computed only on the generated part.
    # Optionally leave one token unmasked for context.
    for i in range(prompt_len - 1):
        labels[i] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def main(config_path):
    print("Loading config...")
    config = load_config(config_path)
    print(f"Loaded config: {config}")

    print("Setting seed...")
    set_seed(config["seed"])
    print("Seed set.")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name_or_path"], token=config["access_token"]
    )
    print("Tokenizer loaded successfully!")

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name_or_path"], token=config["access_token"]
    )
    print("Model loaded successfully!")

    # Ensure pad_token is defined.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # (If you are sure that your tokenizer and model are aligned, you can remove the following check.)
    if len(tokenizer) > model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))

    print("Loading raw datasets...")
    raw_datasets = load_data(
        config["data_dir"], train_fraction=config.get("train_fraction", 1.0)
    )
    print(f"Raw datasets loaded: {raw_datasets}")

    print("Starting dataset tokenization for instruction finetuning...")
    preprocess_fn = lambda x: preprocess_instruction(
        x, tokenizer, config["max_seq_length"]
    )
    tokenized_datasets = raw_datasets.map(preprocess_fn, batched=False)
    print("Datasets tokenized!")

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model, padding="longest", label_pad_token_id=-100
    )

    # Create the output directory if it does not exist.
    os.makedirs(config["output_dir"], exist_ok=True)

    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        num_train_epochs=config["num_train_epochs"],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=os.path.join(config["output_dir"], "logs"),
        logging_steps=100,
        save_total_limit=2,
        seed=config["seed"],
        report_to=["tensorboard"],
        load_best_model_at_end=True,
    )

    print("Creating SFT trainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["val"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()
    print("Training complete!")

    print("Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(config["output_dir"])
    print("All done!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Instruction Finetuning with a config file using SFTTrainer."
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the configuration JSON file.",
    )
    args = parser.parse_args()

    main(args.config_path)
