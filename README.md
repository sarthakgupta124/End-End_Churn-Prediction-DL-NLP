# Churn Detection - Deep Learning Supervised ML

A modular machine learning pipeline for predicting customer churn risk using deep learning with Streamlit UI for interactive predictions.

## Project Structure

```
.
├── app.py                          # Streamlit web interface for predictions
├── requirements.txt                # Project dependencies
├── setup.py                        # Package setup configuration
├── README.md                       # Project documentation
├── SYSTEM_ARCHITECTURE.txt         # System design documentation
├── WORD2VEC_OPTIMIZATION.md        # Word2Vec optimization notes
│
├── notebook/                       # Jupyter notebooks
│   ├── cleaning_modelTraining.ipynb # Data cleaning & model training
│   └── Data/
│       └── datapoints.csv         # Raw dataset
│
├── src/                           # Source code
│   ├── __init__.py
│   ├── logger.py                  # Logging configuration
│   ├── Exception.py               # Custom exception handling
│   ├── utils.py                   # Utility functions (save/load/vectorization)
│   │
│   ├── components/                # Modular pipeline components
│   │   ├── __init__.py
│   │   ├── data_ingestion.py      # Data loading & preparation
│   │   ├── data_transformation.py  # Cleaning, encoding, feature engineering
│   │   └── Model_trainer.py        # Neural network model training
│   │
│   └── pipeline/                  # Orchestration pipelines
│       ├── __init__.py
│       ├── train_pipeline.py       # Training orchestration
│       └── predict_pipeline.py     # Inference pipeline with preprocessing
│
├── artifacts/                     # Generated model & preprocessor artifacts
│   ├── model.h5                   # Trained Keras deep learning model
│   ├── preprocessor.pkl           # OneHotEncoder for categorical features
│   ├── scaler.pkl                 # StandardScaler for numerical features
│   ├── mini_word_dictionary.pkl   # Mini Word2Vec dictionary for feedback
│   ├── ingested_data.csv          # Raw ingested data
│   ├── train.csv                  # Processed training data
│   └── test.csv                   # Processed test data
│
├── logs/                          # Application logs
│   └── log_*.log                  # Timestamped execution logs
│
├── templates/                     # HTML templates (if needed)
├── MLProject.egg-info/            # Package metadata
└── venvChurn/                     # Python virtual environment
```

## Components

### 1. **Data Ingestion** (`src/components/data_ingestion.py`)
- Loads raw customer data from CSV
- Removes unnecessary columns (customer_id, Name, security_no, referral_id)
- Splits into train/test sets for initial data exploration
- Returns cleaned raw dataset path for transformation

### 2. **Data Transformation** (`src/components/data_transformation.py`)
Data cleaning and feature engineering including:
- **Categorical Encoding**:
  - Gender: M/F → 1/0
  - Referral status: Yes/No/? → 1/0 (missing values filled with mode)
  - Complaint status: Multi-class mapping (Solved→1, Unsolved→0, etc.)
  
- **Temporal Features**: Extracts year/month/day from joining_date and hour/min/sec from last_visit_time

- **Text Processing** (Feedback column):
  - Lowercasing and special character removal
  - Stopword removal (English)
  - URL removal
  - Word2Vec embeddings (Google News 300-dimensional)
  - Mini dictionary creation for memory efficiency
  
- **Feature Transformation**:
  - OneHotEncoding for categorical features (region, membership, offers, etc.)
  - StandardScaler for numerical features
  
- **Data Split**: 70% training / 30% testing with stratification

### 3. **Model Training** (`src/components/Model_trainer.py`)
Deep neural network architecture:
- **Input Layer**: 512 neurons (ReLU activation)
- **Batch Normalization** for stable training
- **Hidden Layer 1**: 256 neurons (ReLU) + Dropout(0.3)
- **Hidden Layer 2**: 128 neurons (ReLU) + Dropout(0.2)
- **Output Layer**: 5 neurons (Softmax) for 5-class churn risk classification

Training details:
- **Optimizer**: Adam
- **Loss**: Sparse Categorical Crossentropy
- **Early Stopping**: Patience=8 epochs
- **Evaluation Metrics**: F1-score (macro), Precision, Recall, Classification Report

### 4. **Training Pipeline** (`src/pipeline/train_pipeline.py`)
Orchestrates the complete training workflow:
1. Data Ingestion → Load raw data
2. Data Transformation → Clean, encode, and engineer features
3. Model Training → Build, train, and evaluate neural network
4. Artifact Saving → Store model, preprocessor, scaler, word dictionary

### 5. **Prediction Pipeline** (`src/pipeline/predict_pipeline.py`)
Inference engine for making predictions on new customer data:
- **Artifact Loading**: Loads trained model (model.h5), encoders, scaler, and mini Word2Vec dictionary
- **Data Preprocessing**: Mirrors training-time transformations:
  - Cleans and encodes categorical features using saved preprocessor
  - Vectorizes feedback using mini Word2Vec dictionary
  - Scales numerical features using saved scaler
- **Prediction**: Returns churn risk scores (1-5) and class probabilities
- **Error Handling**: Custom exceptions with detailed logging

### 6. **Streamlit Web UI** (`app.py`)
Interactive web application for real-time predictions:
- Multi-column form with categorized inputs (Profile, Engagement, Behavioral)
- Streamlit caching for efficient artifact loading
- Real-time predictions with confidence scores
- Visual probability distribution chart
- Extended feedback categories for diverse customer feedback

## Installation

### 1. Create Virtual Environment
```bash
conda create -n churn_env python=3.10
conda activate churn_env
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Training Pipeline
Execute the complete training workflow from raw data to model:
```bash
python src/pipeline/train_pipeline.py
```

This will:
1. Load raw data from `notebook/Data/datapoints.csv`
2. Perform data ingestion and save processed data to `artifacts/ingested_data.csv`
3. Clean, encode, and engineer features; create train/test split
4. Train deep learning model with early stopping
5. Save artifacts:
   - `artifacts/model.h5` - Trained Keras model
   - `artifacts/preprocessor.pkl` - OneHotEncoder
   - `artifacts/scaler.pkl` - StandardScaler
   - `artifacts/mini_word_dictionary.pkl` - Word2Vec mini dictionary
6. Generate logs in `logs/` directory

### Web-Based Predictions
Launch the Streamlit application for interactive predictions:
```bash
streamlit run app.py
```

Then:
1. Open browser to `http://localhost:8501`
2. Fill in customer attributes across three columns (Profile, Engagement, Behavioral)
3. Click "Predict Churn Risk" to get real-time prediction
4. View predicted score (1-5) and probability distribution

### Programmatic Predictions
Use the prediction pipeline directly in Python:
```python
import pandas as pd
from src.pipeline.predict_pipeline import PredictionPipeline

# Prepare customer data
customer_data = pd.DataFrame([{
    'age': 45,
    'gender': 'M',
    'region_category': 'City',
    'membership_category': 'Gold Membership',
    'joining_date': '2020-01-15',
    'joined_through_referral': 'Yes',
    'preferred_offer_types': 'Gift Vouchers/Coupons',
    'medium_of_operation': 'Smartphone',
    'internet_option': 'Wi-Fi',
    'last_visit_time': '14:30:45',
    'days_since_last_login': 5,
    'avg_time_spent': 150.5,
    'avg_transaction_value': 25000.0,
    'avg_frequency_login_days': 12.0,
    'points_in_wallet': 650.0,
    'used_special_discount': 'Yes',
    'offer_application_preference': 'Yes',
    'past_complaint': 'No',
    'complaint_status': 'Not Applicable',
    'feedback': 'Excellent service and quality products'
}])

# Make prediction
pipeline = PredictionPipeline()
result = pipeline.predict(customer_data)

print(f"Predicted Churn Risk Score: {result['predicted_scores'][0]}")
print(f"Probabilities: {result['probabilities'][0]}")
```

## Input Features Reference

### Demographic Features:
- **age**: Integer (10-100)
- **gender**: M (Male) or F (Female)
- **region_category**: City, Town, Village
- **membership_category**: No/Basic/Silver/Gold/Platinum/Premium Membership
- **joining_date**: Date in YYYY-MM-DD format

### Account & Referral:
- **joined_through_referral**: Yes, No, or ? (unknown)
- **internet_option**: Wi-Fi, Mobile_Data, Fiber_Optic
- **medium_of_operation**: Desktop, Smartphone, Both, or ?

### Engagement Metrics:
- **days_since_last_login**: Integer (0-365)
- **avg_time_spent**: Minutes (0.0-1000.0)
- **avg_transaction_value**: Currency amount (0.0-100000.0)
- **avg_frequency_login_days**: Days (0.0-31.0)
- **points_in_wallet**: Loyalty points (0.0-2000.0)

### Preferences & Complaints:
- **preferred_offer_types**: Gift Vouchers/Coupons, Credit/Debit Card Offers, Without Offers
- **used_special_discount**: Yes or No
- **offer_application_preference**: Yes or No
- **past_complaint**: Yes or No
- **complaint_status**: Not Applicable, Solved, Solved in Follow-up, Unsolved, No Information Available
- **last_visit_time**: Time in HH:MM:SS format
- **feedback**: Text (processed via Word2Vec)

### Target Variable:
- **churn_risk_score**: 1 (Lowest Risk) to 5 (Highest Risk)

## Output

The model returns predictions in this format:
```json
{
  "predicted_scores": [3],              # 1-5 scale
  "raw_classes": [2],                   # 0-4 internal representation
  "probabilities": [0.1, 0.2, 0.4, 0.2, 0.1]  # Class probabilities
}
```

## Evaluation Metrics

The model is evaluated using:
- **F1-Score (Macro)**: Main evaluation metric across all churn risk classes
- **Precision (Macro)**: Proportion of correct positive predictions
- **Recall (Macro)**: Proportion of actual positives identified
- **Classification Report**: Per-class breakdown for all 5 churn risk scores
- **Early Stopping**: Prevents overfitting using validation loss monitoring

## Key Characteristics

✓ **Modular Architecture**: Components are independent and reusable
✓ **Production-Ready**: Error handling, logging, and artifact management
✓ **Advanced Feature Engineering**: Word2Vec for text, temporal extraction, proper encoding
✓ **Robust Preprocessing**: Handles missing values, categorical encoding, feature scaling
✓ **Deep Learning**: 4-layer neural network with regularization techniques
✓ **Interactive UI**: Streamlit app for non-technical users
✓ **Flexible Inference**: Batch predictions or single-record API
✓ **Reproducible**: Configuration classes for easy tuning
✓ **Scalable Design**: Easy to add new components or modify existing ones

## Requirements

See `requirements.txt` for complete list:
- pandas, numpy
- scikit-learn
- tensorflow/keras
- gensim (for Word2Vec)
- nltk (for stopwords)

## Troubleshooting

### 1. Import Errors
- Run from project root directory: `cd "path/to/ChurnDetection DeepLearning SupervisedML"`
- Activate virtual environment: `conda activate churn_env`
- Install all dependencies: `pip install -r requirements.txt`

### 2. Word2Vec Model Download
- First run may take 5-10 minutes to download Google News Word2Vec (~1.5GB)
- Model is cached locally in gensim's cache directory after first download
- Subsequent runs will be much faster

### 3. Memory Issues  
If out-of-memory errors occur:
- Reduce `batch_size` in `ModelTrainerConfig` (default 32, try 16)
- Reduce `epochs` (default 60, try 30)
- Use smaller portion of data for testing

### 4. Artifact Loading Errors
- Ensure `artifacts/` directory contains all required files:
  - `model.h5`
  - `preprocessor.pkl`
  - `scaler.pkl`
  - `mini_word_dictionary.pkl`
- Run training pipeline first if artifacts are missing: `python src/pipeline/train_pipeline.py`

### 5. Streamlit Port Issues
- If port 8501 is in use, specify different port: `streamlit run app.py --server.port 8502`

## Configuration

### Modify Training Parameters
Edit `ModelTrainerConfig` in `src/components/Model_trainer.py`:
```python
@dataclass
class ModelTrainerConfig:
    epochs: int = 60           # Increase for better accuracy, decrease for speed
    batch_size: int = 32        # Reduce if memory-constrained
    patience: int = 8           # Early stopping patience
```

### Modify Model Architecture
Edit the model layers in `build_model()` method of `ModelTrainer` class

## Future Enhancements

- [ ] Hyperparameter tuning with Optuna
- [ ] K-Fold cross-validation
- [ ] Feature importance visualization (SHAP/LIME)
- [ ] Alternative embeddings (BERT, FastText)
- [ ] FastAPI REST endpoint deployment
- [ ] PostgreSQL integration for prediction logging
- [ ] Advanced analytics dashboard
- [ ] Model versioning and experiment tracking
- [ ] A/B testing framework
- [ ] Real-time model monitoring

## Project Statistics

- **Total Lines of Code**: ~2000+ (components, pipelines, utilities)
- **Data Records**: 36,994 customer records
- **Features**: 21 input features + 300-dim feedback embeddings = 321 total
- **Model Parameters**: ~200K+ trainable parameters
- **Target Classes**: 5 (churn risk scores 1-5)
- **Train/Test Split**: 70/30

## Dependencies Summary

Core packages:
- **TensorFlow/Keras**: Deep learning framework
- **Pandas/NumPy**: Data manipulation
- **Scikit-learn**: Preprocessing and metrics
- **Gensim**: Word2Vec embeddings
- **NLTK**: Natural language processing
- **Streamlit**: Web UI framework
- **Dill**: Object serialization

## License

Open source - feel free to use and modify for your projects.

## Author Notes

This project demonstrates:
- End-to-end ML pipeline development
- Advanced feature engineering (text + temporal)
- Production-ready code structure
- Interactive web deployment
- Error handling and logging best practices
