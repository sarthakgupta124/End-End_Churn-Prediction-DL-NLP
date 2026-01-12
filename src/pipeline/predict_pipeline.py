import os
import re
import sys
import numpy as np
import pandas as pd
import nltk
from dataclasses import dataclass
from tensorflow.keras.models import load_model
from nltk.corpus import stopwords

from src.logger import get_logger
from src.Exception import CustomException
from src.utils import load_object, text_to_vector

# Ensure stopwords are available
try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords")

logger = get_logger(__name__)

@dataclass
class PredictionConfig:
    artifacts_dir: str = "artifacts"
    model_path: str = os.path.join(artifacts_dir, "model.h5")
    preprocessor_path: str = os.path.join(artifacts_dir, "preprocessor.pkl")
    scaler_path: str = os.path.join(artifacts_dir, "scaler.pkl")
    word_dict_path: str = os.path.join(artifacts_dir, "mini_word_dictionary.pkl")

class PredictionPipeline:
    def __init__(self, config: PredictionConfig = None):
        self.config = config or PredictionConfig()
        self.model = None
        self.preprocessor = None
        self.scaler = None
        self.word_dict = None
        self.stop_words = set(stopwords.words("english"))

    def _load_artifacts(self):
        try:
            if self.model is None:
                self.model = load_model(self.config.model_path)
            if self.preprocessor is None:
                self.preprocessor = load_object(self.config.preprocessor_path)
            if self.scaler is None:
                self.scaler = load_object(self.config.scaler_path)
            if self.word_dict is None:
                self.word_dict = load_object(self.config.word_dict_path)
        except Exception as e:
            raise CustomException(e, sys)

    def _clean_input_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies manual mappings to resolve 'string to float' errors"""
        
        # 1. Manual Column Mappings for Binary/Ordinal data
        df['gender'] = df['gender'].map({'M': 1, 'F': 0}).fillna(1)
        df['joined_through_referral'] = df['joined_through_referral'].map({'Yes': 1, 'No': 0, '?': 0}).fillna(0)
        
        binary_cols = ['used_special_discount', 'offer_application_preference', 'past_complaint']
        for col in binary_cols:
            df[col] = df[col].map({'Yes': 1, 'No': 0}).fillna(0)
        
        df['complaint_status'] = df['complaint_status'].map({
            'Not Applicable': 0, 'Solved in Follow-up': 1, 'Unsolved': 0, 
            'Solved': 1, 'No Information Available': 0
        }).fillna(0)

        # 2. Date/Time Features
        df["joining_year"] = df["joining_date"].apply(lambda x: int(str(x).split("-")[0]))
        df["joining_month"] = df["joining_date"].apply(lambda x: int(str(x).split("-")[1]))
        df["joining_day"] = df["joining_date"].apply(lambda x: int(str(x).split("-")[2]))
        
        df["last_visit_hour"] = df["last_visit_time"].apply(lambda x: int(str(x).split(":")[0]))
        df["last_visit_min"] = df["last_visit_time"].apply(lambda x: int(str(x).split(":")[1]))
        df["last_visit_second"] = df["last_visit_time"].apply(lambda x: int(str(x).split(":")[2]))
        
        # 3. Text cleaning
        df["feedback"] = df["feedback"].astype(str).str.lower()
        df["feedback"] = df["feedback"].apply(lambda x: re.sub("[^a-z0-9 ]+", "", x))
        df["feedback"] = df["feedback"].apply(
            lambda x: " ".join([w for w in x.split() if w not in self.stop_words])
        )
        
        df.drop(["joining_date", "last_visit_time"], axis=1, inplace=True)
        return df

    def predict(self, raw_df: pd.DataFrame) -> dict:
        try:
            self._load_artifacts()
            df = raw_df.copy()

            # Step 1: Pre-process and Mappings
            df = self._clean_input_data(df)

            # Step 2: Vectorize Feedback (Keep column to avoid 'missing feedback' error)
            cleaned_feedback = df["feedback"].iloc[0]
            vector = text_to_vector(cleaned_feedback, self.word_dict)
            feedback_cols = [f"feedback_vec_{i}" for i in range(300)]
            feedback_vec_df = pd.DataFrame([vector], columns=feedback_cols)

            # Step 3: Categorical Encoding (Preprocessor needs the 'feedback' column)
            df_encoded_raw = self.preprocessor.transform(df)
            df_encoded = pd.DataFrame(df_encoded_raw, columns=self.preprocessor.get_feature_names_out())

            # Step 4: Drop raw text and merge Word2Vec features
            if 'feedback' in df_encoded.columns:
                df_encoded.drop("feedback", axis=1, inplace=True)
            
            df_final = pd.concat([df_encoded.reset_index(drop=True), feedback_vec_df], axis=1)

            # Step 5: Final Column Alignment for Scaler
            expected_scaler_cols = list(self.scaler.transformers_[0][2])
            df_final = df_final[expected_scaler_cols]

            # Step 6: Inference
            df_scaled = self.scaler.transform(df_final)
            probs = self.model.predict(df_scaled, verbose=0)
            score_index = np.argmax(probs, axis=1)[0]
            
            # Use keys that match app.py exactly
            return {
                "score": int(score_index + 1), # App expects "score"
                "probabilities": probs[0].tolist() # App expects "probabilities"
            }

        except Exception as e:
            logger.error(f"Prediction Pipeline Error: {e}")
            raise CustomException(e, sys)