import os
import json
from datasets import Dataset, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import RewardTrainer, RewardConfig, get_peft_config


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


# TODO: Find a better way for generating rejected response
def generate_preference_pair(example):
    """
    Generate an implicit preference pair from a PLOS article entry.
    The "summary" field is used as the chosen (preferred) response.
    A simple heuristic is used to create a rejected response by removing the last sentence.
    """
    summary = example.get("summary", "")
    if isinstance(summary, list):
        summary_text = " ".join(summary).strip()
    else:
        summary_text = summary.strip()
    sentences = [s.strip() for s in summary_text.split(".") if s.strip()]
    if len(sentences) > 1:
        rejected_text = ". ".join(sentences[:-1]).strip() + "."
    else:
        rejected_text = summary_text
    return {"chosen": summary_text, "rejected": rejected_text}


# TODO: Return MeSH terms for each article
def preprocess_preference_example(example, tokenizer, max_length):
    """
    Converts an example into tokenized candidate responses.
    Returns a dict with keys:
      - "input_ids_chosen", "attention_mask_chosen"
      - "input_ids_rejected", "attention_mask_rejected"
    """
    pair = generate_preference_pair(example)
    tokenized_chosen = tokenizer(
        pair["chosen"],
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )
    tokenized_rejected = tokenizer(
        pair["rejected"],
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )
    return {
        "input_ids_chosen": tokenized_chosen["input_ids"],
        "attention_mask_chosen": tokenized_chosen["attention_mask"],
        "input_ids_rejected": tokenized_rejected["input_ids"],
        "attention_mask_rejected": tokenized_rejected["attention_mask"],
    }


if __name__ == "__main__":
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

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name_or_path"], token=config["access_token"]
    )
    print("Loading finetuned model from:", config["finetuned_model_dir"])
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
    preprocess_fn = lambda x: preprocess_preference_example(
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

    print("Initializing RewardTrainer...")
    # TODO: Override the compute_rewards method
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

    print("Saving reward-finetuned model to:", reward_training_args.output_dir)
    trainer.save_model(reward_training_args.output_dir)
    tokenizer.save_pretrained(reward_training_args.output_dir)
    print("Reward modeling complete!")
