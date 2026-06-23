# Databricks notebook source
# %pip install telco_churn-0.1.0-py3-none-any.whl
# COMMAND ----------

# MAGIC %md
# MAGIC ## Send Data to the Endpoint

# COMMAND ----------

from pyspark.sql.functions import col
from pyspark.sql import SparkSession
import os
from pyspark.dbutils import DBUtils

from telco_churn.config import ProjectConfig
from telco_churn.utils import is_databricks
from databricks.sdk import WorkspaceClient
from dotenv import load_dotenv

spark = SparkSession.builder.getOrCreate()

# Load configuration
config = ProjectConfig.from_yaml(config_path="../project_config_telco.yml", env="dev")

test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set") \
                        .withColumn("customerID", col("customerID").cast("string")) \
                        .toPandas()

# COMMAND ----------

from databricks.sdk import WorkspaceClient
import requests
import time

workspace = WorkspaceClient()

# Required columns for inference (Telco Churn)
required_columns = [
    "MaleGender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "Tenure",
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


# COMMAND ----------

def send_request_workspace(dataframe_record):
    """
    Sends a request to the model serving endpoint using workspace client.
    """
    response = workspace.serving_endpoints.query(
        name="telco-churn-model-serving",
        dataframe_records=dataframe_record,
    )
    return response

# COMMAND ----------

# Sample records for testing
sampled_records = test_set[required_columns].sample(n=100, replace=True).to_dict(orient="records")
dataframe_records = [[record] for record in sampled_records]

# COMMAND ----------

# Test the endpoint
for record in dataframe_records:
    response = send_request_workspace(record)
    print(response)
    

# COMMAND ----------

from databricks.connect import DatabricksSession
from databricks.sdk import WorkspaceClient

from telco_churn.config import ProjectConfig
from telco_churn.monitoring import create_or_refresh_monitoring

spark = DatabricksSession.builder.getOrCreate()
workspace = WorkspaceClient()

# Load configuration
config = ProjectConfig.from_yaml(config_path="../project_config_telco.yml", env="dev")

create_or_refresh_monitoring(config=config, spark=spark, workspace=workspace)

# COMMAND ----------
