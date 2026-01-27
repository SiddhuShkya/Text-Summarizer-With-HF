from src.textSummarizer.logging import logger
from src.textSummarizer.pipeline.stage1_data_ingestion import (
    DataIngestionTrainingPipeline,
)
from src.textSummarizer.pipeline.stage2_data_transformation import (
    DataTransformationTrainingPipeline,
)

STAGE_NAME = "Data Ingestion Stage"

try:
    logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.initiate_data_ingestion()
    logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<")
except Exception as e:
    logger.exception(f"Error in stage {STAGE_NAME}: {e}")
    raise e

STAGE_NAME = "Data Transformation Stage"

try:
    logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
    from src.textSummarizer.pipeline.stage2_data_transformation import (
        DataTransformationTrainingPipeline,
    )

    data_transformation = DataTransformationTrainingPipeline()
    data_transformation.initiate_data_transformation()
    logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<")
except Exception as e:
    logger.exception(f"Error in stage {STAGE_NAME}: {e}")
    raise e
