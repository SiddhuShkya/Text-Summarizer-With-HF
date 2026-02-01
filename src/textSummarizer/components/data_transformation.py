import os
import pandas as pd
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict
from src.textSummarizer.logging import logger
from src.textSummarizer.entity import DataTransformationConfig


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name)
        logger.info(f"Initialized DataTransformation with tokenizer: {self.config.tokenizer_name}")

    def convert_examples_to_features(self, example_batch):
        model_inputs = self.tokenizer(
            example_batch["dialogue"],  # Tokenize input dialogue
            text_target=example_batch["summary"],  # Tokenize target summary for seq2seq
            max_length=1024,  # Max length for input tokens
            truncation=True,  # Truncate if longer than max_length
        )
        labels = self.tokenizer(
            text_target=example_batch["summary"],
            max_length=128,
            truncation=True,  # Tokenize summary as labels
        )
        model_inputs["labels"] = labels[
            "input_ids"
        ]  # Add tokenized labels to model inputs
        return model_inputs

    def convert(self):
        dataset_splits = {}

        for split in ["train", "validation", "test"]:
            csv_path = os.path.join(self.config.data_path, f"{split}.csv")

            df = pd.read_csv(csv_path)
            ds = Dataset.from_pandas(df)

            ds = ds.map(self.convert_examples_to_features, batched=True)

            dataset_splits[split] = ds

        summarizer_dataset = DatasetDict(dataset_splits)
        logger.info("Saving transformed dataset to disk...")
        summarizer_dataset.save_to_disk(
            os.path.join(self.config.root_dir, "summarizer_dataset")
        )
        logger.info(f"Dataset saved successfully at {os.path.join(self.config.root_dir, 'summarizer_dataset')}")
