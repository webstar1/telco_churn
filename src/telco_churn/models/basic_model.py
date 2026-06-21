"""Basic model implementation for Marvel character classification.

infer_signature (from mlflow.models) → Captures input-output schema for model tracking.

num_features → List of numerical feature names.
cat_features → List of categorical feature names.
target → The column to predict (Alive).
parameters → Hyperparameters for LightGBM.
catalog_name, schema_name → Database schema names for Databricks tables.
"""

import mlflow
from delta.tables import DeltaTable
from loguru import logger
from mlflow import MlflowClient
from mlflow.models import infer_signature
from pyspark.sql import SparkSession
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from telco_churn.config import ProjectConfig, Tags


class BasicModel:
    """A basic model class for telco churn prediction using random forests.

    This class handles data loading, feature preparation, model training, and MLflow logging.
    """

    def __init__(self, config: ProjectConfig, tags: Tags, spark: SparkSession) -> None:
        """Initialize the model with project configuration.

        :param config: Project configuration object
        :param tags: Tags object
        :param spark: SparkSession object
        """
        self.config = config
        self.spark = spark

        # Extract settings from the config
        self.num_features = self.config.num_features
        self.cat_features = self.config.cat_features
        self.target = self.config.target
        self.parameters = self.config.parameters
        self.catalog_name = self.config.catalog_name
        self.schema_name = self.config.schema_name
        self.experiment_name = self.config.experiment_name_basic
        self.model_name = f"{self.catalog_name}.{self.schema_name}.telco_churn_model_basic"
        self.tags = tags.to_dict()

    def load_data(self) -> None:
        """Load training and testing data from Delta tables.

        Splits data into features (X_train, X_test) and target (y_train, y_test).
        """
        logger.info("🔄 Loading data from Databricks tables...")
        self.train_set_spark = self.spark.table(f"{self.catalog_name}.{self.schema_name}.train_set")
        self.train_set = self.train_set_spark.toPandas()
        self.test_set_spark = self.spark.table(f"{self.catalog_name}.{self.schema_name}.test_set")
        self.test_set = self.test_set_spark.toPandas()

        self.X_train = self.train_set[self.num_features]
        self.y_train = self.train_set[self.target]
        self.X_test = self.test_set[self.num_features]
        self.y_test = self.test_set[self.target]
        self.eval_data = self.test_set[self.num_features + [self.target]]

        train_delta_table = DeltaTable.forName(
            self.spark, f"{self.catalog_name}.{self.schema_name}.train_set"
        )
        self.train_data_version = str(train_delta_table.history().select("version").first()[0])
        test_delta_table = DeltaTable.forName(
            self.spark, f"{self.catalog_name}.{self.schema_name}.test_set"
        )
        self.test_data_version = str(test_delta_table.history().select("version").first()[0])
        logger.info("✅ Data successfully loaded.")

    def prepare_features(self) -> None:
        """Constructs a pipeline combining preprocessing and random forest classification model."""
        logger.info("🔄 Defining preprocessing pipeline...")

        self.pipeline = Pipeline(steps=[("model", RandomForestClassifier(**self.parameters))])
        logger.info("✅ Preprocessing pipeline defined.")

    def train(self) -> None:
        """Train the model."""
        logger.info("🚀 Starting training...")
        self.pipeline.fit(self.X_train, self.y_train)

    def log_model(self) -> None:
        mlflow.set_experiment(self.experiment_name)

        with mlflow.start_run(tags=self.tags) as run:
            self.run_id = run.info.run_id

            signature = infer_signature(
                model_input=self.X_train, model_output=self.pipeline.predict_proba(self.X_train)
            )

            mlflow.log_input(
                mlflow.data.from_spark(
                    self.train_set_spark,
                    table_name=f"{self.catalog_name}.{self.schema_name}.train_set",
                    version=self.train_data_version,
                ),
                context="training",
            )

            self.model_info = mlflow.sklearn.log_model(
                sk_model=self.pipeline,
                artifact_path="random-forest-model",
                signature=signature,
                input_example=self.X_test[0:1],
            )

            # SAFE evaluation (no Spark / no registry dependency)
            from sklearn.metrics import f1_score

            preds = self.pipeline.predict(self.X_test)
            self.metrics = {"f1_score": f1_score(self.y_test, preds)}

    def model_improved(self) -> bool:
        """Evaluate the model performance on the test set.

        Compares the current model with the latest registered model using F1-score.
        :return: True if the current model performs better, False otherwise.
        """
        client = MlflowClient()
        try:
            latest_model_version = client.get_model_version_by_alias(
                name=self.model_name, alias="latest-model"
            )
            latest_model_uri = f"models:/{latest_model_version.model_id}"

            result = mlflow.models.evaluate(
                latest_model_uri,
                self.eval_data,
                targets=self.config.target,
                model_type="classifier",
                evaluators=["default"],
            )
            metrics_old = result.metrics
            if self.metrics["f1_score"] >= metrics_old["f1_score"]:
                logger.info("Current model performs better. Returning True.")
                return True
            else:
                logger.info("Current model does not improve over latest. Returning False.")
                return False
        except Exception as e:
            logger.info(
                f"No previous model found or evaluation failed: {e}. Treating as first run."
            )
            return True

    def register_model(self) -> None:
        """Register model in Unity Catalog."""
        logger.info("🔄 Registering the model in UC...")
        registered_model = mlflow.register_model(
            model_uri=f"runs:/{self.run_id}/random-forest-model",
            name=self.model_name,
            tags=self.tags,
        )
        logger.info(f"✅ Model registered as version {registered_model.version}.")

        latest_version = registered_model.version

        client = MlflowClient()
        client.set_registered_model_alias(
            name=self.model_name,
            alias="latest-model",
            version=latest_version,
        )
        return latest_version
