import os
import pandas as pd
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict
from src.textSummarizer.entity import DataTransformationConfig


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name)

    def convert_examples_to_features(self, example_batch):
        inputs = self.tokenizer(
            example_batch["dialogue"],
            max_length=512,
            truncation=True,
        )

        with self.tokenizer.as_target_tokenizer():
            targets = self.tokenizer(
                example_batch["summary"],
                max_length=128,
                truncation=True,
            )

        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": targets["input_ids"],
        }

    def convert(self):
        dataset_splits = {}

        for split in ["train", "validation", "test"]:
            csv_path = os.path.join(self.config.data_path, f"{split}.csv")

            df = pd.read_csv(csv_path)
            ds = Dataset.from_pandas(df)

            ds = ds.map(
                self.convert_examples_to_features,
                batched=True,
                remove_columns=ds.column_names,
            )

            dataset_splits[split] = ds

        summarizer_dataset = DatasetDict(dataset_splits)

        summarizer_dataset.save_to_disk(
            os.path.join(self.config.root_dir, "summarizer_dataset")
        )
