"""
Submit Hyperparameter Tuning Job to Vertex AI - Using Python Package
Run this script from your Vertex AI notebook or local machine with gcloud configured.
"""

from google.cloud import aiplatform
from google.cloud.aiplatform import hyperparameter_tuning as hpt

# Configuration
PROJECT_ID = "three-little-pigs-476102"
REGION = "asia-southeast1"
BUCKET_NAME = "bucket-three-little-pigs-476102"

X_TRAIN_PATH = f"gs://{BUCKET_NAME}/train.csv" 
X_TEST_PATH = f"gs://{BUCKET_NAME}/test.csv"   
Y_TRAIN_PATH = f"gs://{BUCKET_NAME}/y_train.csv"
Y_TEST_PATH = f"gs://{BUCKET_NAME}/y_test.csv"

# Job settings
DISPLAY_NAME = "insurance-xgboost-hpt-v3"

# Initialize Vertex AI
aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=f"gs://{BUCKET_NAME}")

# Define hyperparameter search space
hyperparameter_spec = {
    'max-depth': hpt.IntegerParameterSpec(min=3, max=10, scale='linear'),
    'learning-rate': hpt.DoubleParameterSpec(min=0.01, max=0.3, scale='log'),
    'n-estimators': hpt.IntegerParameterSpec(min=50, max=500, scale='linear'),
    'min-child-weight': hpt.IntegerParameterSpec(min=1, max=10, scale='linear'),
    'subsample': hpt.DoubleParameterSpec(min=0.6, max=1.0, scale='linear'),
    'colsample-bytree': hpt.DoubleParameterSpec(min=0.6, max=1.0, scale='linear'),
    'gamma': hpt.DoubleParameterSpec(min=0, max=5, scale='linear'),
    'reg-alpha': hpt.DoubleParameterSpec(min=0.0001, max=10, scale='log'),
    'reg-lambda': hpt.DoubleParameterSpec(min=0.1, max=10, scale='log'),
}

# Metric spec
metric_spec = {
    'rmse': 'minimize',
}

# Create worker pool spec using Python package
worker_pool_specs = [{
    'machine_spec': {
        'machine_type': 'n1-standard-4',
    },
    'replica_count': 1,
    'python_package_spec': {
        'executor_image_uri': 'us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-11.py310:latest',
        'package_uris': [f'gs://{BUCKET_NAME}/trainer.tar.gz'],
        'python_module': 'trainer.trainer',
        'args': [
            f'--x-train-path={X_TRAIN_PATH}',
            f'--x-test-path={X_TEST_PATH}',
            f'--y-train-path={Y_TRAIN_PATH}',
            f'--y-test-path={Y_TEST_PATH}',
        ],
    },
}]

# Define the custom training job
custom_job = aiplatform.CustomJob(
    display_name=DISPLAY_NAME,
    worker_pool_specs=worker_pool_specs,
)

# Create hyperparameter tuning job
hp_job = aiplatform.HyperparameterTuningJob(
    display_name=DISPLAY_NAME,
    custom_job=custom_job,
    metric_spec=metric_spec,
    parameter_spec=hyperparameter_spec,
    max_trial_count=30,
    parallel_trial_count=5,
    search_algorithm='random',
)

print(f"Starting hyperparameter tuning job: {DISPLAY_NAME}")
print(f"X_train path: {X_TRAIN_PATH}")
print(f"X_test path: {X_TEST_PATH}")
print(f"y_train path: {Y_TRAIN_PATH}")
print(f"y_test path: {Y_TEST_PATH}")
print(f"Max trials: 30, Parallel trials: 5")
print(f"Using package: gs://{BUCKET_NAME}/trainer.tar.gz")

# Submit the job
hp_job.run()

print("\n" + "="*50)
print("Hyperparameter tuning job completed!")
print(f"Best trial: {hp_job.trials[0]}")
print("\nBest hyperparameters:")
for param, value in hp_job.trials[0].parameters.items():
    print(f"  {param}: {value}")