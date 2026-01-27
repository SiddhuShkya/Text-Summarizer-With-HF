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
