import torch
import evaluate
import pandas as pd
from datasets import load_from_disk
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from src.textSummarizer.logging import logger
from src.textSummarizer.entity import ModelEvaluationConfig


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initialized ModelEvaluation on device: {self.device}")
        self.rouge_metric = evaluate.load("rouge")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_path).to(
            self.device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        self.dataset = load_from_disk(self.config.data_path)
        logger.info(f"Loaded model from {self.config.model_path} and dataset from {self.config.data_path}")

    def generate_batch_sized_chunks(self, list_of_elements, batch_size):
        for i in range(0, len(list_of_elements), batch_size):
            yield list_of_elements[i : i + batch_size]

    def calculate_metric_on_test_ds(
        self,
        dataset,
        metric,
        model,
        tokenizer,
        device,
        batch_size=1,
        column_text="article",
        column_summary="highlights",
    ):
        model.eval()  # Set model to evaluation mode

        # Split dataset into batches
        article_batches = list(
            self.generate_batch_sized_chunks(dataset[column_text], batch_size)
        )
        target_batches = list(
            self.generate_batch_sized_chunks(dataset[column_summary], batch_size)
        )

        with torch.no_grad():  # Disable gradient calculation
            for article_batch, target_batch in zip(article_batches, target_batches):
                inputs = tokenizer(
                    article_batch,
                    max_length=256,  # Max input length
                    truncation=True,  # Truncate if too long
                    padding="max_length",  # Pad to max length
                    return_tensors="pt",  # Return PyTorch tensors
                )

                summaries = model.generate(
                    input_ids=inputs["input_ids"].to(device),  # Move inputs to device
                    attention_mask=inputs["attention_mask"].to(device),
                    max_new_tokens=128,  # Max tokens for output
                    num_beams=1,  # Beam search width
                    do_sample=False,  # Deterministic generation
                    use_cache=True,  # Use past key values for speed
                )

                decoded_summaries = tokenizer.batch_decode(
                    summaries,
                    skip_special_tokens=True,  # Decode outputs to text
                )

                metric.add_batch(
                    predictions=decoded_summaries,  # Add generated summaries
                    references=target_batch,  # Add reference summaries
                )

        return metric.compute()  # Compute final metric (e.g., ROUGE, BLEU)

    def evaluate(self):
        dataset = load_from_disk(self.config.data_path)
        rouge_metric = self.rouge_metric
        rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
        score = self.calculate_metric_on_test_ds(
            dataset=dataset["validation"],
            metric=rouge_metric,
            device=self.device,
            model=self.model,
            tokenizer=self.tokenizer,
            batch_size=2,
            column_text="dialogue",
            column_summary="summary",
        )
        rouge_dict = {name: score[name] for name in rouge_names}
        df = pd.DataFrame(rouge_dict, index=["flan-t5-small-finetuned"])
        logger.info("Saving evaluation metrics to CSV...")
        df.to_csv(self.config.metric_file_name, index=True)
        logger.info(f"Evaluation metrics saved successfully at {self.config.metric_file_name}")
