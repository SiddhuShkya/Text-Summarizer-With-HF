from transformers import pipeline
from transformers import AutoTokenizer
from src.textSummarizer.config.configuration import ConfigurationManager


class PredictionPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_model_evaluation_config()

    def predict(self, text: str) -> str:
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        model = self.config.model_path
        gen_kwargs = {"length_penalty": 0.8, "num_beams": 8, "max_length": 128}

        pipe = pipeline("summarization", model=model, tokenizer=tokenizer)
        output = pipe(text, **gen_kwargs)[0]["summary_text"]
        print(f"Summarized Text: {output}")
        return output
