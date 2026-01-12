import os
import sys
import pandas as pd
from dataclasses import dataclass
from src.logger import get_logger
from src.Exception import CustomException

logger = get_logger(__name__)


@dataclass
class DataIngestionConfig:
    """Configuration for data ingestion"""
    raw_data_path: str = 'notebook/Data/datapoints.csv'
    artifacts_dir: str = 'artifacts'
    ingested_data_path: str = os.path.join(artifacts_dir, 'ingested_data.csv')


class DataIngestion:
    
    def __init__(self, config: DataIngestionConfig = None):
        self.config = config or DataIngestionConfig()
    
    def load_data(self) -> pd.DataFrame:
        try:
            logger.info(f"Loading data from {self.config.raw_data_path}")
            df = pd.read_csv(self.config.raw_data_path)
            logger.info(f"Data shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error in load_data: {str(e)}")
            raise CustomException(e, sys)
    
    def drop_unnecessary_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            columns_to_drop = ['customer_id', 'Name', 'security_no', 'referral_id'] 
            df = df.drop(columns=columns_to_drop, axis=1)
            
            
            logger.info(f"Data shape after dropping columns: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error in drop_unnecessary_columns: {str(e)}")
            raise CustomException(e, sys)
    
    def initiate_data_ingestion(self) -> str:
        try:
            logger.info("Starting data ingestion")
            
            # Load raw data
            df = self.load_data()
            
            # Drop unnecessary columns
            newdata = self.drop_unnecessary_columns(df)
            
            # Save ingested data
            newdata.to_csv(self.config.ingested_data_path, index=False,header=True)
            
            logger.info("Data ingestion completed successfully")
            return self.config.ingested_data_path
        except Exception as e:
            logger.error(f"Error in initiate_data_ingestion: {str(e)}")
            raise CustomException(e, sys)
