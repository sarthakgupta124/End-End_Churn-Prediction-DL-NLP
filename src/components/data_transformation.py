"""
Data Transformation Component - Data cleaning and feature engineering
"""

import sys
import os
import re
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import gensim.downloader as api
import nltk
from nltk.corpus import stopwords

from src.logger import get_logger
from src.Exception import CustomException
from src.utils import save_object

logger = get_logger(__name__)

# Download stopwords if not already present
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')


@dataclass
class DataTransformationConfig:
    artifacts_dir: str = 'artifacts'
    preprocessor_path: str = os.path.join(artifacts_dir, 'preprocessor.pkl')
    scaler_path: str = os.path.join(artifacts_dir, 'scaler.pkl')
    train_path: str = os.path.join(artifacts_dir, 'train.csv')
    test_path: str = os.path.join(artifacts_dir, 'test.csv')

    word2vec_model: str ='word2vec-google-news-300'

class DataTransformation:
    """Data transformation component for cleaning and feature engineering"""
    
    def __init__(self, config: DataTransformationConfig = None):
        self.config = DataTransformationConfig()
        self.ohe = OneHotEncoder(drop='first')
        self.scaler = StandardScaler()
        self.wv = None
        
    def _load_word2vec_model(self):
        """Load pre-trained Word2Vec model"""
        try:
            if self.wv is None:
                logger.info(f"Loading {self.config.word2vec_model} model...")
                self.wv = api.load(self.config.word2vec_model)
                logger.info("Word2Vec model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading Word2Vec model: {str(e)}")
            raise CustomException(e, sys)
    
    def clean_gender(self, df: pd.DataFrame):
        """Clean gender column"""
        try:
            df['gender']=df['gender'].map({'M':1,'F':0})
            df['gender'].fillna(df['gender'].mode()[0], inplace=True)
            logger.info("Gender column cleaned")
            return df
        except Exception as e:
            logger.error(f"Error cleaning gender: {str(e)}")
            raise
    
    def clean_referral(self, df: pd.DataFrame):
        """Clean referral column"""
        try:
            df["joined_through_referral"]=df["joined_through_referral"].map({'Yes':1,'No':0,'?':np.nan})
            df['joined_through_referral'].fillna(df['joined_through_referral'].mode()[0],inplace=True)
            logger.info("Referral column cleaned")
            return df
        except Exception as e:
            logger.error(f"Error cleaning referral: {str(e)}")
            raise
    
    def clean_region_category(self, df: pd.DataFrame):
        """Clean region category column"""
        try:
            df['region_category'].fillna(df['region_category'].mode()[0],inplace=True)
            logger.info("Region category column cleaned")
            return df
        except Exception as e:
            logger.error(f"Error cleaning region_category: {str(e)}")
            raise
    
    def extract_date_features(self, df: pd.DataFrame):
        """Extract year, month, day from joining_date"""
        try:
            df["joining_year"]=df["joining_date"].apply(lambda x: int(x.split('-')[0]))
            df["joining_month"]=df["joining_date"].apply(lambda x: int(x.split('-')[1]))
            df["joining_day"]=df["joining_date"].apply(lambda x: int(x.split('-')[2]))
            df.drop(['joining_date'], axis=1, inplace=True)
            logger.info("Date features extracted")
            return df
        except Exception as e:
            logger.error(f"Error extracting date features: {str(e)}")
            raise
    
    def clean_offer_types(self, df: pd.DataFrame):
        """Clean preferred offer types"""
        try:
            df['preferred_offer_types'].fillna(df['preferred_offer_types'].mode()[0],inplace=True)
            logger.info("Offer types column cleaned")
            return df
        except Exception as e:
            logger.error(f"Error cleaning offer types: {str(e)}")
            raise
    
    def clean_medium_operation(self, df: pd.DataFrame):
        """Clean medium of operation"""
        try:
            df['medium_of_operation'].replace({"?":df['medium_of_operation'].mode()[0]}, inplace=True)
            logger.info("Medium of operation cleaned")
            return df
        except Exception as e:
            logger.error(f"Error cleaning medium_operation: {str(e)}")
            raise
    
    def extract_time_features(self, df: pd.DataFrame):
        """Extract hour, minute, second from last_visit_time"""
        try:
            df["last_visit_hour"]=df["last_visit_time"].apply(lambda x: int(x.split(':')[0]))
            df["last_visit_min"]=df["last_visit_time"].apply(lambda x: int(x.split(':')[1]))
            df["last_visit_second"]=df["last_visit_time"].apply(lambda x: int(x.split(':')[2]))
            df.drop(['last_visit_time'], axis=1, inplace=True)
            logger.info("Time features extracted")
            return df
        except Exception as e:
            logger.error(f"Error extracting time features: {str(e)}")
            raise
    
    def clean_numeric_columns(self, df: pd.DataFrame):
        """Clean numeric columns with missing values"""
        try:
            df['avg_frequency_login_days'] = pd.to_numeric(df['avg_frequency_login_days'], errors='coerce').astype(float)
            df = df[df['points_in_wallet'].notna()]
            logger.info("Numeric columns cleaned")
            return df
        except Exception as e:
            logger.error(f"Error cleaning numeric columns: {str(e)}")
            raise
    
    def encode_binary_columns(self, df: pd.DataFrame):
        """Encode binary columns (Yes/No)"""
        try:
            binary_cols = ['used_special_discount', 'offer_application_preference', 'past_complaint']
            for col in binary_cols:
                    df[col] = df[col].map({'Yes': 1, 'No': 0})
            
            # Clean complaint status
            df['complaint_status']=df['complaint_status'].map({'Not Applicable':0,'Solved in Follow-up':1,'Unsolved':0,'Solved':1,'No Information Available':np.nan})
            df['complaint_status'].fillna(df['complaint_status'].mode()[0],inplace=True)
            
            logger.info("Binary columns encoded")
            return df
        except Exception as e:
            logger.error(f"Error encoding binary columns: {str(e)}")
            raise
    
    def clean_feedback_text(self, data: pd.DataFrame):
        """Clean feedback text"""
        try:
            logger.info("Processing feedback text...")
            data['feedback']=data['feedback'].apply(lambda x:re.sub('[^a-z A-z 0-9-]+', '',x)) 
            ## Remove the stopswords
            data['feedback']=data['feedback'].apply(lambda x:" ".join([y for y in x.split() if y not in stopwords.words('english')]))
            ## Remove url 
            data['feedback']=data['feedback'].apply(lambda x: re.sub(r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', '' , str(x)))
            ## Remove any additional spaces
            data['feedback']=data['feedback'].apply(lambda x: " ".join(x.split()))
            
            logger.info("Feedback text cleaned successfully")
            return data  # CRITICAL: This was missing!
        except Exception as e:
            logger.error(f"Error cleaning feedback: {str(e)}")
            return data
    
    def encode_categorical_features(self, df_train: pd.DataFrame,columns_name:list ):
        
        try:
            
            ct = ColumnTransformer([('onehot', self.ohe, columns_name)], remainder='passthrough',verbose_feature_names_out=False)
            
            _transformed = ct.fit_transform(df_train)
            col_n=ct.get_feature_names_out()
            n_data= pd.DataFrame(_transformed, columns=col_n)
            return n_data, ct
        except Exception as e:
            logger.error(f"Error in encode_categorical_features: {str(e)}")
            raise CustomException(e, sys)
    
    def scale_features(self, df_train: pd.DataFrame,columns_name:list):
        """Scale features using StandardScaler"""
        try:
            st = ColumnTransformer([('scaler', self.scaler, columns_name)], remainder='passthrough',verbose_feature_names_out=False)
            
            _transformed = st.fit_transform(df_train)
            col_n=st.get_feature_names_out()
            n_data= pd.DataFrame(_transformed, columns=col_n)
            return n_data, st
        except Exception as e:
            logger.error(f"Error in scale_features: {str(e)}")
            raise CustomException(e, sys)
    
    def process_feedback_column(self, df_: pd.DataFrame):
        
    
        try:
            if self.wv is None:
                self._load_word2vec_model()

            
            logger.info("Processing feedback column...")
            def text_to_vector(text, model):
                words = text.split()
                word_vectors = []
                
                for word in words:
                    if word in model:
                        word_vectors.append(model[word])
                
                if len(word_vectors) == 0:
                    return np.zeros(300)  # Return zero vector if no words found
                
                return np.mean(word_vectors, axis=0)
            logger.info("Applying Word2Vec transformation...")
            df__feedback_vectors = df_['feedback'].apply(lambda x: text_to_vector(x, self.wv))
            feedback_train_df = pd.DataFrame(df__feedback_vectors.tolist(), 
                                  columns=[f'feedback_vec_{i}' for i in range(300)],
                                  index=df_.index)
            # Create and save mini word dictionary
            unique_words = set(" ".join(df_['feedback']).split())

            # 2. Extract only those vectors from the big model
            mini_word_vectors = {}
            for word in unique_words:
                if word in self.wv:
                    mini_word_vectors[word] = self.wv[word]

            save_object('artifacts/mini_word_dictionary.pkl', mini_word_vectors)
            
            df_ = df_.drop('feedback', axis=1)
            df_ = pd.concat([df_, feedback_train_df], axis=1)


            
            
            return df_

        except Exception as e:
            logger.error(f"Error processing feedback: {str(e)}")
            raise CustomException(e, sys)
        
    
    def initiate_data_transformation(self, data_path: str):
        
        try:
            logger.info("Starting data transformation")
            
            # Load data
            df = pd.read_csv(data_path)
            logger.info(f"Data loaded with shape: {df.shape}")
            
            # Data cleaning
            df = self.clean_gender(df)
            df = self.clean_referral(df)
            df = self.clean_region_category(df)
            df = self.extract_date_features(df)
            df = self.clean_offer_types(df)
            df = self.clean_medium_operation(df)
            df = self.extract_time_features(df)
            df = self.clean_numeric_columns(df)
            df = self.encode_binary_columns(df)



            X=df.drop('churn_risk_score',axis=1)
            y=df['churn_risk_score']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            cat_cols=['region_category','membership_category','preferred_offer_types','medium_of_operation','internet_option']
            x_train,encoder_cat=self.encode_categorical_features(X_train,cat_cols)
            x_test_transformed=encoder_cat.transform(X_test)
            columns=encoder_cat.get_feature_names_out()
            x_test=pd.DataFrame(x_test_transformed,columns=columns)


            for col in x_train.columns:
                if x_train[col].dtype=='object' and col!='feedback':
                    if x_train[col].isna().sum()>0:
                        x_train[col].fillna(x_train[col].mode()[0],inplace=True)


            for col in x_test.columns:
                if x_test[col].dtype=='object' and col!='feedback':
                    if x_test[col].isna().sum()>0:
                        x_test[col].fillna(x_test[col].mode()[0],inplace=True)   

            for col in x_train.columns:
                if x_train[col].dtype=='object' and col!='feedback':
                    x_train[col]=x_train[col].astype(float)
                    x_test[col]=x_test[col].astype(float) 



            x_train['feedback']=x_train['feedback'].str.lower()
            x_test['feedback']=x_test['feedback'].str.lower()


            x_train=self.clean_feedback_text(x_train)
            x_test=self.clean_feedback_text(x_test)


            x_train=self.process_feedback_column(x_train)
            x_test=self.process_feedback_column(x_test)
            

            col_standard=list(x_train.columns)
            x_train,encoder_stnd=self.scale_features(x_train,col_standard)
            x_test_transformed=encoder_stnd.transform(x_test)
            columns=encoder_stnd.get_feature_names_out()
            x_test=pd.DataFrame(x_test_transformed,columns=col_standard)


            # 1. Fix invalid scores (-1) by clipping them to the 1-5 range
            y_train = y_train.clip(1, 5)
            y_test = y_test.clip(1, 5)

            # 2. Shift labels from 1-5 to 0-4 for the Softmax layer
            y_train_final = y_train - 1
            y_test_final = y_test - 1


            
            
            # Save transformers
            save_object(self.config.preprocessor_path, encoder_cat)
            save_object(self.config.scaler_path, encoder_stnd)
            
            # Save datasets
           
            y_train_final.name = 'churn_risk_score'
            y_test_final.name = 'churn_risk_score'

          
            x_train = x_train.reset_index(drop=True)
            y_train_final = y_train_final.reset_index(drop=True)

            x_test = x_test.reset_index(drop=True)
            y_test_final = y_test_final.reset_index(drop=True)

            train_set = pd.concat([x_train, y_train_final], axis=1)
            test_set = pd.concat([x_test, y_test_final], axis=1)

            # 4. Save to CSV
            train_set.to_csv(self.config.train_path, index=False)
            test_set.to_csv(self.config.test_path, index=False)

            
            logger.info("Data transformation completed successfully")
            
            return (self.config.train_path, self.config.test_path)
        
        except Exception as e:
            logger.error(f"Error in initiate_data_transformation: {str(e)}")
            raise CustomException(e, sys)
