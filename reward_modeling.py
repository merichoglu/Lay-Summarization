import os
import json
import warnings
import torch
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, set_seed
from trl import RewardTrainer, RewardConfig, ScriptArguments, get_peft_config


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_data(data_dir, train_fraction=1.0):
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
            print(f"Using {num_samples} samples from training.")
        datasets[split] = Dataset.from_list(data)
    return DatasetDict(datasets)


def reward_function(generated_summary, mesh_terms):
    """
    A simple reward function that returns a score based on the presence of MeSH terms.
    For each MeSH term present in the generated summary, add a fixed reward.
    """
    reward = 0.0
    for term in mesh_terms:
        if term.lower() in generated_summary.lower():
            reward += 1.0
    return reward


def preprocess_for_reward(example, tokenizer, max_length):
    """
    Prepares an example for reward training.
    This function builds a prompt from title, abstract, and mesh_terms,
    then appends the summary. If the summary is a list, it is joined into a single string.
    """
    instruction = "generate a lay summary of the following scientific article."
    title = example.get("title", "").strip()
    abstract_val = example.get("abstract", "")
    abstract = (
        " ".join(abstract_val)
        if isinstance(abstract_val, list)
        else abstract_val.strip()
    )
    mesh_terms = example.get("mesh_terms", [])
    input_field = (
        f"Title: {title}\nAbstract: {abstract}\nMeSH Terms: {', '.join(mesh_terms)}\n"
    )
    prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_field}\n\n### Response:\n"

    # Handle summary being a list or a string
    summary = example.get("summary", "")
    if isinstance(summary, list):
        summary = " ".join(summary)

    full_text = prompt + summary

    tokenized_full = tokenizer(
        full_text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )
    tokenized_prompt = tokenizer(
        prompt,
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )
    input_ids = tokenized_full["input_ids"]
    attention_mask = tokenized_full["attention_mask"]
    prompt_len = sum(tokenized_prompt["attention_mask"])
    labels = tokenized_full["input_ids"].copy()
    for i in range(prompt_len - 1):
        labels[i] = -100
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "prompt": prompt,
        "mesh_terms": mesh_terms,
    }


if __name__ == "__main__":
    # Load configuration
    import argparse

    parser = argparse.ArgumentParser(
        description="Reward Modeling for Lay Summarization"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to reward modeling config JSON file",
    )
    args = parser.parse_args()
    config = load_config(args.config_path)

    set_seed(config["seed"])

    # Load tokenizer and finetuned model.
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name_or_path"], token=config["access_token"]
    )
    model = AutoModelForCausalLM.from_pretrained(
        config["finetuned_model_dir"], token=config["access_token"]
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading dataset...")
    raw_datasets = load_data(
        config["data_dir"], train_fraction=config.get("train_fraction", 1.0)
    )

    print("Preprocessing dataset for reward modeling...")
    preprocess_fn = lambda x: preprocess_for_reward(
        x, tokenizer, config["max_seq_length"]
    )
    tokenized_datasets = raw_datasets.map(preprocess_fn, batched=False)

    # Create a RewardConfig for training.
    reward_training_args = RewardConfig(
        output_dir=config["output_dir"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        num_train_epochs=config["num_train_epochs"],
        logging_dir=os.path.join(config["output_dir"], "logs"),
        logging_steps=config["logging_steps"],
        eval_steps=config["eval_steps"],
        max_length=config["max_seq_length"],
        seed=config["seed"],
        remove_unused_columns=False,
        center_rewards_coefficient=config.get("center_rewards_coefficient", None),
    )

    peft_config = None
    if config.get("use_peft", False):
        peft_config = get_peft_config(config)

    trainer = RewardTrainer(
        model=model,
        processing_class=tokenizer,
        args=reward_training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=(
            tokenized_datasets["val"]
            if reward_training_args.eval_strategy != "no"
            else None
        ),
        peft_config=peft_config,
    )

    print("Starting reward model fine-tuning...")
    trainer.train()

    print("Saving reward-finetuned model...")
    trainer.save_model(reward_training_args.output_dir)
    tokenizer.save_pretrained(reward_training_args.output_dir)
    print("Reward modeling complete!")
