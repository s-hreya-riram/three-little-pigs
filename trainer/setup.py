from setuptools import find_packages, setup

setup(
    name='insurance-xgboost-trainer',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'pandas>=1.3.0',
        'numpy>=1.21.0',
        'scikit-learn>=1.0.0',
        'xgboost>=1.5.0',
        'cloudml-hypertune>=0.1.0.dev6',
        'google-cloud-storage>=2.0.0',
    ],
    python_requires='>=3.7',
    description='XGBoost hyperparameter tuning for insurance dataset'
)
