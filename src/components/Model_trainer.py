"""
Model Training Component - Build and train neural network model
"""

import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

from src.logger import get_logger
from src.Exception import CustomException

logger = get_logger(__name__)


@dataclass
class ModelTrainerConfig:
    """Configuration for model training"""
    artifacts_dir: str = 'artifacts'
    model_path: str = os.path.join(artifacts_dir, 'model.h5')
    
    # Model parameters
    input_neurons: int = 512
    hidden_neurons_1: int = 256
    hidden_neurons_2: int = 128
    output_neurons: int = 5
    dropout_1: float = 0.3
    dropout_2: float = 0.2
    dropout_3: float = 0.2
    
    # Training parameters
    epochs: int = 60
    batch_size: int = 32
    patience: int = 8
    optimizer: str = 'adam'
    loss: str = 'sparse_categorical_crossentropy'


class ModelTrainer:
    """Model training component for building and training the neural network"""
    
    def __init__(self, config: ModelTrainerConfig = None):
        """
        Initialize ModelTrainer
        
        Args:
            config: ModelTrainerConfig object
        """
        self.config = ModelTrainerConfig()
        self.model = None
        self.history = None
    
    def build_model(self, input_shape: int):
        try:
            logger.info(f"Building model with input shape: {input_shape}")
            
            self.model = Sequential([
                # Input Layer
                Dense(self.config.input_neurons, activation='relu', input_shape=(input_shape,)),
                BatchNormalization(),
                Dropout(self.config.dropout_1),
                
                # Hidden Layers
                Dense(self.config.hidden_neurons_1, activation='relu'),
                BatchNormalization(),
                Dropout(self.config.dropout_2),
                
                Dense(self.config.hidden_neurons_2, activation='relu'),
                Dropout(self.config.dropout_3),
                
                # Output Layer
                Dense(self.config.output_neurons, activation='softmax')
            ])
            
            # Compile model
            self.model.compile(
                optimizer=self.config.optimizer,
                loss=self.config.loss,
                metrics=['accuracy']
            )
            
            logger.info("Model built and compiled successfully")
            logger.info(self.model.summary())
            
            return self.model
        except Exception as e:
            logger.error(f"Error building model: {str(e)}")
            raise CustomException(e, sys)
    
    def train_model(self, X_train: pd.DataFrame, y_train: pd.DataFrame,
                   X_test: pd.DataFrame, y_test: pd.DataFrame):
        try:
            logger.info("Starting model training")
            
            # Early stopping callback
            early_stop = EarlyStopping(
                monitor='val_loss',
                patience=self.config.patience,
                restore_best_weights=True
            )
            
            # Train model
            self.history = self.model.fit(
                X_train, y_train,
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                validation_data=(X_test, y_test),
                callbacks=[early_stop],
                verbose=1
            )
            
            logger.info("Model training completed successfully")
            return self.history
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise CustomException(e, sys)
    
    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.DataFrame):
        try:
            logger.info("Evaluating model on test data")
            
            # 1. Get the raw probability predictions
            y_pred_probs = self.model.predict(X_test)

            # 2. Convert probabilities to class labels (0-4)
            y_pred = np.argmax(y_pred_probs, axis=1)

            
            # Note: y_test_final is your shifted 0-4 ground truth
            score = 100 * f1_score(y_test, y_pred, average="macro")
            print(f"Evaluation Score: {score:.2f}")

            # 4. Detailed breakdown (Precision, Recall, F1 for each class)
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, target_names=['Score 1', 'Score 2', 'Score 3', 'Score 4', 'Score 5']))
            

            recall = recall_score(y_test, y_pred, average='macro')
            precision = precision_score(y_test, y_pred, average='macro')
            print(f"Recall: {recall:.4f}, Precision: {precision:.4f}")
            return {
                'f1_score': score,
                'recall': recall,
                'precision': precision,
                'y_pred': y_pred,
                'y_pred_probs': y_pred_probs
            }
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            raise CustomException(e, sys)
    
    def save_model(self):
        """Save the trained model"""
        try:
            os.makedirs(os.path.dirname(self.config.model_path), exist_ok=True)
            self.model.save(self.config.model_path)
            logger.info(f"Model saved to {self.config.model_path}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise CustomException(e, sys)
    
    def initiate_model_trainer(self,train_path: str,test_path: str):
        
        try:
            logger.info("Starting model training pipeline")
            
            # Load data
            
            train = pd.read_csv(train_path)
            test = pd.read_csv(test_path)
            
            x_train=train.drop(columns=['churn_risk_score'],axis=1)
            y_train = train['churn_risk_score']
            x_test=test.drop(columns=['churn_risk_score'],axis=1)
            y_test = test['churn_risk_score']
            
            logger.info(f"Data loaded. X_train shape: {x_train.shape}, X_test shape: {x_test.shape}")
            
            # Build model
            self.build_model(x_train.shape[1])
            
            # Train model
            self.train_model(x_train, y_train, x_test, y_test)
            
            # Evaluate model
            results = self.evaluate_model(x_test, y_test)
            
            # Save model
            self.save_model()
            
            logger.info("Model training pipeline completed successfully")
            
            return {
                'model': self.model,
                'results': results,
                'model_path': self.config.model_path
            }
        except Exception as e:
            logger.error(f"Error in initiate_model_trainer: {str(e)}")
            raise CustomException(e, sys)
