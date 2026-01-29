from src.textSummarizer.logging import logger
from src.textSummarizer.pipeline.stage1_data_ingestion import (
    DataIngestionTrainingPipeline,
)
from src.textSummarizer.pipeline.stage2_data_transformation import (
    DataTransformationTrainingPipeline,
)
from src.textSummarizer.pipeline.stage3_model_trainer import (
    ModelTrainerTrainingPipeline,
)
from src.textSummarizer.pipeline.stage4_model_evaluation import (
    ModelEvaluationTrainingPipeline,
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
    data_transformation = DataTransformationTrainingPipeline()
    data_transformation.initiate_data_transformation()
    logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<")
except Exception as e:
    logger.exception(f"Error in stage {STAGE_NAME}: {e}")
    raise e

STAGE_NAME = "Model Trainer Stage"

try:
    logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
    model_trainer = ModelTrainerTrainingPipeline()
    model_trainer.initiate_model_trainer()
    logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<")
except Exception as e:
    logger.exception(f"Error in stage {STAGE_NAME}: {e}")
    raise e

STAGE_NAME = "Model Evaluation Stage"

try:
    logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
    model_evaluation = ModelEvaluationTrainingPipeline()
    model_evaluation.initiate_model_evaluation()
    logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<")
except Exception as e:
    logger.exception(f"Error in stage {STAGE_NAME}: {e}")
    raise e
