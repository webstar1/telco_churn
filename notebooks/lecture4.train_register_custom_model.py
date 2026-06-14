# Databricks notebook source
import mlflow
from pyspark.sql import SparkSession

from telco_churn.config import ProjectConfig, Tags
from telco_churn.models.custom_model import TelcoChurnModelWrapper
from importlib.metadata import version
from dotenv import load_dotenv
from mlflow import MlflowClient
import os

# Set up Databricks or local MLflow tracking
def is_databricks():
    return "DATABRICKS_RUNTIME_VERSION" in os.environ

# COMMAND ----------

# If you have DEFAULT profile and are logged in with DEFAULT profile,
# skip these lines

if not is_databricks():
    mlflow.set_tracking_uri("databricks://Zabs")
    mlflow.set_registry_uri("databricks-uc://Zabs")


config = ProjectConfig.from_yaml(config_path="../project_config_telco.yml", env="dev")
spark = SparkSession.builder.getOrCreate()
tags = Tags(**{"git_sha": "4c1e4ba45402fd415e8189fccd9ce291077357b2", "branch": "main"})
telco_churn_v = version("telco_churn")

code_paths=[f"../dist/telco_churn-{telco_churn_v}-py3-none-any.whl"]

# COMMAND ----------

client = MlflowClient()
wrapped_model_version = client.get_model_version_by_alias(
    name=f"{config.catalog_name}.{config.schema_name}.telco_churn_model_basic",
    alias="latest-model")
# Initialize model with the config path

# COMMAND ----------

test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").toPandas()
X_test = test_set[config.num_features]

# COMMAND ----------

pyfunc_model_name = f"{config.catalog_name}.{config.schema_name}.telco_churn_model_custom"
wrapper = TelcoChurnModelWrapper()
wrapper.log_register_model(wrapped_model_uri=f"models:/{wrapped_model_version.name}/{wrapped_model_version.version}",
                           pyfunc_model_name=pyfunc_model_name,
                           experiment_name=config.experiment_name_custom,
                           input_example=X_test[0:1],
                           tags=tags,
                           code_paths=code_paths)

# COMMAND ----------

# unwrap and predict
loaded_pufunc_model = mlflow.pyfunc.load_model(f"models:/{pyfunc_model_name}@latest-model")

unwraped_model = loaded_pufunc_model.unwrap_python_model()

# COMMAND ----------

unwraped_model.predict(context=None, model_input=X_test[0:1])

# COMMAND ----------
latest = client.get_model_version_by_alias(
    name=pyfunc_model_name,
    alias="latest-model"
)
print(f"Latest custom model version: {latest.version}")
print(f"Run ID: {latest.run_id}")

# COMMAND ----------

# another predict function with uri

loaded_pufunc_model.predict(X_test[0:1])