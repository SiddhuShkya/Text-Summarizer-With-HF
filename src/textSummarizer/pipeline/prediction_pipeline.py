from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.textSummarizer.config.configuration import ConfigurationManager
from src.textSummarizer.logging import logger


class PredictionPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_model_evaluation_config()

    def predict(self, text: str) -> str:
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
            model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_path)
            gen_kwargs = {
                "max_length": 20,
                "min_length": 5,
                "length_penalty": 2.0,
                "num_beams": 4,
            }
            pipe = pipeline("summarization", model=model, tokenizer=tokenizer)
            output = pipe(text, **gen_kwargs)[0]["summary_text"]
            print(f"Summarized Text: {output}")
            return output
        except Exception as e:
            logger.exception(e)
            raise e
