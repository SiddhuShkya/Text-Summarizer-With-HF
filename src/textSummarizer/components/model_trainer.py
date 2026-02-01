import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForSeq2Seq
from datasets import load_from_disk
from src.textSummarizer.logging import logger
from src.textSummarizer.entity import ModelTrainerConfig


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initialized ModelTrainer on device: {self.device}")

    def train(self):
        # Load dataset
        dataset = load_from_disk(self.config.data_path)
        # Load model and tokenizer
        model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt).to(
            self.device
        )
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
        # Data collator
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.root_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            weight_decay=self.config.weight_decay,
            logging_steps=self.config.logging_steps,
            eval_strategy=self.config.eval_strategy,
            warmup_steps=self.config.warmup_steps,
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
        )
        # Initialize Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["test"],
            eval_dataset=dataset["validation"],
            data_collator=data_collator,
            processing_class=tokenizer,
        )
        # Start training
        logger.info("Starting model training...")
        trainer.train()
        logger.info("Model training completed successfully.")
        # Save the model and tokenizer
        logger.info("Saving model and tokenizer...")
        model.save_pretrained(os.path.join(self.config.root_dir, "t5_model"))
        tokenizer.save_pretrained(os.path.join(self.config.root_dir, "t5_tokenizer"))
        logger.info(f"Model and tokenizer saved at {self.config.root_dir}")
