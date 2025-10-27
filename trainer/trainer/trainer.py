"""
Vertex AI Hyperparameter Tuning Script for XGBoost
This script trains an XGBoost model and reports metrics for hyperparameter optimization.
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import hypertune
import logging
import json
from datetime import datetime
from google.cloud import storage

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--x-train-path', type=str, default='gs://bucket-three-little-pigs-476102/train.csv')
    parser.add_argument('--x-test-path', type=str, default='gs://bucket-three-little-pigs-476102/test.csv')
    parser.add_argument('--y-train-path', type=str, default='gs://bucket-three-little-pigs-476102/y_train.csv')
    parser.add_argument('--y-test-path', type=str, default='gs://bucket-three-little-pigs-476102/y_test.csv')
    parser.add_argument('--max-depth', type=int, default=6)
    parser.add_argument('--learning-rate', type=float, default=0.3)
    parser.add_argument('--n-estimators', type=int, default=100)
    parser.add_argument('--min-child-weight', type=int, default=1)
    parser.add_argument('--subsample', type=float, default=1.0)
    parser.add_argument('--colsample-bytree', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=0)
    parser.add_argument('--reg-alpha', type=float, default=0)
    parser.add_argument('--reg-lambda', type=float, default=1)
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument('--output-bucket', type=str, default='bucket-three-little-pigs-476102')
    parser.add_argument('--output-folder', type=str, default='hpt_results')
    return parser.parse_args()


def load_data(x_train_path, x_test_path, y_train_path, y_test_path):
    """Load pre-split data from GCS."""
    logger.info("Loading data from GCS...")
    
    X_train = pd.read_csv(x_train_path)
    X_test = pd.read_csv(x_test_path)
    y_train = pd.read_csv(y_train_path)
    y_test = pd.read_csv(y_test_path)
  
    if y_train.shape[1] == 1:
        y_train = y_train.iloc[:, 0]
        y_test = y_test.iloc[:, 0]
      
    logger.info(f"X_train: {X_train.shape[0]} rows, {X_train.shape[1]} features")
    logger.info(f"X_test: {X_test.shape[0]} rows, {X_test.shape[1]} features")
    logger.info(f"y_train: {len(y_train)} samples")
    logger.info(f"y_test: {len(y_test)} samples")
    
    return X_train, X_test, y_train, y_test

def preprocess_features(X_train, X_test):
    for col in list(X_train.columns):
        if 'Unnamed' in str(col):
            logger.info(f"Dropping column: {col}")
            X_train = X_train.drop(columns=[col])
            X_test = X_test.drop(columns=[col])
    return X_train, X_test


def train_model(X_train, y_train, X_test, y_test, args):
    """Train XGBoost model with given hyperparameters."""
    logger.info("Training XGBoost model...")
    
    # Configure XGBoost parameters
    params = {
        'max_depth': args.max_depth,
        'learning_rate': args.learning_rate,
        'n_estimators': args.n_estimators,
        'min_child_weight': args.min_child_weight,
        'subsample': args.subsample,
        'colsample_bytree': args.colsample_bytree,
        'gamma': args.gamma,
        'reg_alpha': args.reg_alpha,
        'reg_lambda': args.reg_lambda,
        'objective': 'reg:squarederror',
        'random_state': args.random_state,
        'n_jobs': -1
    }
    
    logger.info(f"Hyperparameters: {params}")
    
    # Train model
    model = xgb.XGBRegressor(**params)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        early_stopping_rounds=10,
        verbose=False
    )
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    logger.info(f"Train RMSE: {train_rmse:.4f}")
    logger.info(f"Test RMSE: {test_rmse:.4f}")
    logger.info(f"Train R²: {train_r2:.4f}")
    logger.info(f"Test R²: {test_r2:.4f}")
    logger.info(f"Test MAE: {test_mae:.4f}")
    
    return model, test_rmse, test_r2, train_rmse, train_r2, test_mae


def save_results_to_gcs(args, test_rmse, test_r2, train_rmse, train_r2, test_mae):
    """Save trial results to GCS bucket."""
    try:
        # Create timestamp for unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Prepare results dictionary
        results = {
            'timestamp': timestamp,
            'hyperparameters': {
                'max_depth': args.max_depth,
                'learning_rate': args.learning_rate,
                'n_estimators': args.n_estimators,
                'min_child_weight': args.min_child_weight,
                'subsample': args.subsample,
                'colsample_bytree': args.colsample_bytree,
                'gamma': args.gamma,
                'reg_alpha': args.reg_alpha,
                'reg_lambda': args.reg_lambda,
            },
            'metrics': {
                'test_rmse': float(test_rmse),
                'test_r2': float(test_r2),
                'test_mae': float(test_mae),
                'train_rmse': float(train_rmse),
                'train_r2': float(train_r2),
            }
        }
        
        # Initialize GCS client
        storage_client = storage.Client()
        bucket = storage_client.bucket(args.output_bucket)
        
        # Save as JSON
        json_blob = bucket.blob(f'{args.output_folder}/trial_{timestamp}.json')
        json_blob.upload_from_string(
            json.dumps(results, indent=2),
            content_type='application/json'
        )
        
        # Save as TXT (human-readable)
        txt_content = f"""
================================================================================
HYPERPARAMETER TUNING TRIAL RESULTS
================================================================================
Timestamp: {timestamp}

HYPERPARAMETERS:
--------------------------------------------------------------------------------
max_depth:          {args.max_depth}
learning_rate:      {args.learning_rate}
n_estimators:       {args.n_estimators}
min_child_weight:   {args.min_child_weight}
subsample:          {args.subsample}
colsample_bytree:   {args.colsample_bytree}
gamma:              {args.gamma}
reg_alpha:          {args.reg_alpha}
reg_lambda:         {args.reg_lambda}

METRICS:
--------------------------------------------------------------------------------
Test RMSE:          {test_rmse:.6f}
Test R²:            {test_r2:.6f}
Test MAE:           {test_mae:.6f}
Train RMSE:         {train_rmse:.6f}
Train R²:           {train_r2:.6f}

================================================================================
"""
        
        txt_blob = bucket.blob(f'{args.output_folder}/trial_{timestamp}.txt')
        txt_blob.upload_from_string(txt_content, content_type='text/plain')
        
        logger.info(f"Results saved to gs://{args.output_bucket}/{args.output_folder}/trial_{timestamp}.*")
        
    except Exception as e:
        logger.warning(f"Failed to save results to GCS: {str(e)}")
        logger.warning("Continuing without saving results...")


def main():
    """Main training function."""
    args = get_args()
    
    # Load pre-split data
    X_train, X_test, y_train, y_test = load_data(
        args.x_train_path,
        args.x_test_path,
        args.y_train_path,
        args.y_test_path
    )
    
    # Preprocess features if needed
    X_train, X_test = preprocess_features(X_train, X_test)
    
    # Train model
    model, test_rmse, test_r2, train_rmse, train_r2, test_mae = train_model(
        X_train, y_train, X_test, y_test, args
    )
    
    # Save results to GCS
    save_results_to_gcs(args, test_rmse, test_r2, train_rmse, train_r2, test_mae)
   
    hpt = hypertune.HyperTune()
    hpt.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag='rmse',  # Must match config
        metric_value=test_rmse,
        global_step=1
    )
    
    # Also report R² for reference
    hpt.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag='r2',
        metric_value=test_r2,
        global_step=1
    )
    # ============================================================================
    
    logger.info("Training completed successfully!")


if __name__ == '__main__':
    main()