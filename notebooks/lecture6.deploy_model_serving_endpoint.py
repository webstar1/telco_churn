# Databricks notebook source
# MAGIC %pip install telco_churn-1.0.1-py3-none-any.whl

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import time
import os
import requests
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession
from mlflow import mlflow
from databricks.sdk import WorkspaceClient
from dotenv import load_dotenv

from telco_churn.config import ProjectConfig
from telco_churn.serving.model_serving import ModelServing
from telco_churn.utils import is_databricks


# COMMAND ----------

# spark session
spark = SparkSession.builder.getOrCreate()

w = WorkspaceClient()

os.environ["DBR_HOST"] = w.config.host
os.environ["DBR_TOKEN"] = w.tokens.create(lifetime_seconds=1200).token_value

if not is_databricks():
    load_dotenv()
    profile = os.environ["PROFILE"]
    mlflow.set_tracking_uri(f"databricks://{profile}")
    mlflow.set_registry_uri(f"databricks-uc://{profile}")

# Load project config
config = ProjectConfig.from_yaml(config_path="../project_config_telco.yml", env="dev")
catalog_name = config.catalog_name
schema_name = config.schema_name

# COMMAND ----------

# Initialize model serving
model_serving = ModelServing(
    model_name=f"{catalog_name}.{schema_name}.telco_churn_model_custom", endpoint_name="telco-churn-model-serving"
)

# COMMAND ----------

# Deploy the model serving endpoint
model_serving.deploy_or_update_serving_endpoint()


# COMMAND ----------

# Create a sample request body
required_columns = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges"
]

# Load test data
test_set = spark.table(
    f"{config.catalog_name}.{config.schema_name}.test_set"
).toPandas()

# Sample records
sampled_records = (
    test_set[required_columns]
    .sample(n=1000, replace=True)
)

# Replace NaN values with None
import numpy as np

sampled_records = (
    sampled_records
    .replace({np.nan: None})
    .to_dict(orient="records")
)

dataframe_records = [[record] for record in sampled_records]

# COMMAND ----------

"""
Each dataframe record should contain the encoded features used by the model.

Example:

[{
    'gender': 1,
    'SeniorCitizen': 0,
    'Partner': 1,
    'Dependents': 0,
    'tenure': 24,
    'PhoneService': 1,
    'Contract': 2,
    'PaperlessBilling': 1,
    'MonthlyCharges': 75.25,
    'TotalCharges': 1806.00
}]
"""

def call_endpoint(record):
    """
    Calls the model serving endpoint with a given input record.
    """
    serving_endpoint = f"{os.environ['DBR_HOST']}/serving-endpoints/telco-churn-model-serving/invocations"    
    print(f"Calling endpoint: {serving_endpoint}")
    
    response = requests.post(
        serving_endpoint,
        headers={"Authorization": f"Bearer {os.environ['DBR_TOKEN']}"},
        json={"dataframe_records": record},
    )
    return response.status_code, response.text


status_code, response_text = call_endpoint(dataframe_records[0])
print(f"Response Status: {status_code}")
print(f"Response Text: {response_text}")

# COMMAND ----------

# Load test
for i in range(len(dataframe_records)):
    status_code, response_text = call_endpoint(dataframe_records[i])
    print(f"Response Status: {status_code}")
    print(f"Response Text: {response_text}")
    time.sleep(0.2) 
