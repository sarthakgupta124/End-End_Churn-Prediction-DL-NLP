"""
Training Pipeline - Orchestrate the entire training workflow
"""

import sys
from src.logger import get_logger
from src.Exception import CustomException
from src.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.Model_trainer import ModelTrainer, ModelTrainerConfig

logger = get_logger(__name__)


class TrainingPipeline:
    """Training pipeline to orchestrate the entire ML workflow"""
    
    def __init__(self):
        """Initialize training pipeline"""
        self.data_ingestion_config = DataIngestionConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_config = ModelTrainerConfig()
    
    def run(self):
        """Execute the complete training pipeline"""
        try:
            logger.info("="*60)
            logger.info("Starting Training Pipeline")
            logger.info("="*60)
            
            # Step 1: Data Ingestion
            logger.info("\nStep 1: Data Ingestion")
            logger.info("-" * 40)
            data_ingestion = DataIngestion(self.data_ingestion_config)
            ingested_data_path = data_ingestion.initiate_data_ingestion()
            logger.info(f" Data ingestion completed: {ingested_data_path}")
            
            # Step 2: Data Transformation
            logger.info("\nStep 2: Data Transformation")
            logger.info("-" * 40)
            data_transformation = DataTransformation(self.data_transformation_config)
            train_path, test_path=data_transformation.initiate_data_transformation(ingested_data_path)
            logger.info(f" Data transformation completed")
            logger.info(f"  - Train data: {train_path}")
            logger.info(f"  - Test data: {test_path}")
            
            # Step 3: Model Training
            logger.info("\nStep 3: Model Training")
            logger.info("-" * 40)
            model_trainer = ModelTrainer(self.model_trainer_config)
            training_results = model_trainer.initiate_model_trainer(
                train_path, test_path
            )
            logger.info(f" Model training completed")
            logger.info(f"  - Model saved: {training_results['model_path']}")
            
            logger.info("\n" + "="*60)
            logger.info("Training Pipeline Completed Successfully!")
            logger.info("="*60)
            
            return training_results
        
        except Exception as e:
            logger.error(f"Error in training pipeline: {str(e)}")
            raise CustomException(e, sys)


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    results = pipeline.run()
