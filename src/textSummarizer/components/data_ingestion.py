import os
import zipfile
import urllib.request as request
from src.textSummarizer.logging import logger
from src.textSummarizer.entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_data(self):
        if not os.path.exists(self.config.local_data_file):
            file_name, headers = request.urlretrieve(
                url=self.config.source_url, filename=self.config.local_data_file
            )
            logger.info("File downloaded successfully!")
        else:
            logger.info(
                f"File already exists at {self.config.local_data_file}. Skipping download."
            )

    def extract_zip_file(self):
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, "r") as zip_ref:
            zip_ref.extractall(unzip_path)
        logger.info(f"File extracted successfully at {unzip_path}")
