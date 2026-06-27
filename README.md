# Telco Churn Prediction Model by Zohaib

A machine learning pipeline for predicting customer churn in telecommunications using Databricks and MLOps best practices.

## Set up your environment

This project uses UV for Python dependency management. Check out the documentation on how to install it: https://docs.astral.sh/uv/getting-started/installation/

To create a new environment and install dependencies, run:

```bash
uv sync --extra dev
```

## Project Overview

This project demonstrates end-to-end machine learning operations (MLOps) including data preprocessing, model training, experiment tracking, model deployment, A/B testing, and monitoring on Databricks.

## Data

The Telco Churn dataset contains customer information including:
- Demographics (age, tenure, contracts)
- Services used (internet, phone, streaming)
- Billing information
- Churn status (target variable)

This data is used to build a classification model that predicts whether a customer will churn.

## Notebooks

The pipeline consists of the following notebooks:

- **01_data_preprocessing.py**: Load, clean, and preprocess the Telco dataset. Split into train/test sets and save to the catalog.
- **02_mlflow_experiment_tracking.py**: Set up experiment tracking and log baseline models using MLflow.
- **03_train_register_basic_model.py**: Train a basic classification model and register it in the MLflow Model Registry.
- **04_train_register_custom_model.py**: Train an advanced custom model with feature engineering and hyperparameter tuning.
- **05_ab_testing.py**: Implement A/B testing framework to compare model versions in production.
- **06_deploy_model_serving_endpoint.py**: Deploy the champion model to a Databricks model serving endpoint.
- **07_create_monitoring_table.py**: Set up monitoring and drift detection for the deployed model.

## Project Structure

```
├── notebooks/              # Interactive notebooks for each pipeline stage
├── src/                    # Python source code and utilities
├── scripts/                # Automation and deployment scripts
├── data/                   # Data directory (gitignored)
├── tests/                  # Unit tests
├── resources/              # Configuration and resource files
├── databricks.yml          # Databricks workspace configuration
├── project_config_telco.yml # Project-specific configuration
└── pyproject.toml          # Python project dependencies
```

## Running the Pipeline

Execute the notebooks in order from `01_` to `07_` to run the complete ML pipeline from data preprocessing through model monitoring.
