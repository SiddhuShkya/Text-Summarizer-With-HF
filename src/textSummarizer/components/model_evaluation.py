import torch
import evaluate
import pandas as pd
from datasets import load_from_disk
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from src.textSummarizer.entity import ModelEvaluationConfig


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.rouge_metric = evaluate.load("rouge")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_path).to(
            self.device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        self.dataset = load_from_disk(self.config.data_path)

    def generate_batch_sized_chunks(self, list_of_elements, batch_size):
        for i in range(0, len(list_of_elements), batch_size):
            yield list_of_elements[i : i + batch_size]

    def calculate_metric_on_test_ds(
        self,
        dataset,
        metric,
        device,
        model,
        tokenizer,
        batch_size=1,
        column_text="article",
        column_summary="highlights",
    ):
        model.eval()

        article_batches = list(
            self.generate_batch_sized_chunks(dataset[column_text], batch_size)
        )
        target_batches = list(
            self.generate_batch_sized_chunks(dataset[column_summary], batch_size)
        )

        with torch.no_grad():
            for article_batch, target_batch in zip(article_batches, target_batches):
                inputs = tokenizer(
                    article_batch,
                    max_length=256,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )

                summaries = model.generate(
                    input_ids=inputs["input_ids"].to(device),
                    attention_mask=inputs["attention_mask"].to(device),
                    max_new_tokens=128,
                    num_beams=1,
                    do_sample=False,
                    use_cache=True,
                )

                decoded_summaries = tokenizer.batch_decode(
                    summaries, skip_special_tokens=True
                )

                metric.add_batch(
                    predictions=decoded_summaries,
                    references=target_batch,
                )

        return metric.compute()

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
        df = pd.DataFrame(rouge_dict, index=[f"pegasus"])
        df.to_csv(self.config.metric_file_name, index=False)
