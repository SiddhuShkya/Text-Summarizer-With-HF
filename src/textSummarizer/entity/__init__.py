from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataIngestionConfig:
    root_dir: str
    source_url: str
    local_data_file: str
    unzip_dir: str


@dataclass
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    tokenizer_name: Path


@dataclass
class ModelTrainerConfig:
    # From config.yaml
    root_dir: Path
    data_path: Path
    model_ckpt: Path
    # From params.yaml
    num_train_epochs: int
    warmup_steps: int
    per_device_train_batch_size: int
    weight_decay: float
    logging_steps: int
    eval_strategy: str
    eval_steps: int
    save_steps: float
    gradient_accumulation_steps: int


@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    data_path: Path
    model_path: Path
    tokenizer_path: Path
    metric_file_name: Path
