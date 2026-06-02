"""Data preprocessing module for Telco Churn."""

import time

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from sklearn.model_selection import train_test_split

from telco_churn.config import ProjectConfig


class DataProcessor:
    """A class for preprocessing and managing Telco Churn DataFrame operations.

    This class handles data preprocessing, splitting, and saving to Databricks tables.
    """

    def __init__(self, pandas_df: pd.DataFrame, config: ProjectConfig, spark: SparkSession) -> None:
        self.df = pandas_df  # Store the DataFrame as self.df
        self.config = config  # Store the configuration
        self.spark = spark
        self.target_encodings = {}

    def preprocess(self) -> None:
        """Preprocess the Telco Churn DataFrame stored in self.df.

        This method handles missing values, converts data types, and performs feature engineering.
        """
        cat_features = self.config.cat_features
        num_features = self.config.num_features
        target = self.config.target

        self.df.rename(columns={"gender": "MaleGender"}, inplace=True)
        self.df.rename(columns={"tenure": "Tenure"}, inplace=True)

        # Total Charges
        contract_map = {
            'One year': 12,
            'Two year': 24
        }
        self.df["TotalCharges"] = self.df["TotalCharges"].replace(" ", np.nan)
        self.df["TotalCharges"] = self.df["TotalCharges"].astype(float)
        self.df['TotalCharges'] = self.df['TotalCharges'].fillna(self.df['MonthlyCharges'] * self.df['Contract'].map(contract_map))

        # Gender
        self.df['MaleGender'] = self.df['MaleGender'].map({'Male': 1, 'Female': 0})

        # Multiple Lines
        self.df['MultipleLines'] = self.df['MultipleLines'].replace('No phone service', 'No')

        # Convert internet columns to flag columns
        internet_cols = [
            'OnlineSecurity',
            'OnlineBackup',
            'DeviceProtection',
            'TechSupport',
            'StreamingTV',
            'StreamingMovies'
        ]

        self.df[internet_cols] = self.df[internet_cols].replace('No internet service', 'No')

        # Convert flag columns to numeric
        binary_cols = ['Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'Churn']

        for col in binary_cols:
            self.df[col] = self.df[col].map({'Yes': 1, 'No': 0})

    def split_data(self, test_size: float = 0.2, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split the DataFrame (self.df) into training and test sets.

        :param test_size: The proportion of the dataset to include in the test split.
        :param random_state: Controls the shuffling applied to the data before applying the split.
        :return: A tuple containing the training and test DataFrames.
        """
        train_set, test_set = train_test_split(self.df, test_size=test_size, random_state=random_state)
        return train_set, test_set

    def fit_target_encoding(self, train_df) -> None:

        target_encode_cols = [
            'InternetService',
            'Contract',
            'PaymentMethod'
        ]

        for col in target_encode_cols:

            self.target_encodings[col] = (
                train_df.groupby(col)['Churn']
                .mean()
                .to_dict()
            )


    def apply_target_encoding(self,train_df,test_df):

        target_encode_cols = [
            'InternetService',
            'Contract',
            'PaymentMethod'
        ]

        for col in target_encode_cols:

            train_df[col] = (
                train_df[col]
                .map(self.target_encodings[col])
            )

            test_df[col] = (
                test_df[col]
                .map(self.target_encodings[col])
            )

        return train_df, test_df

    def save_to_catalog(self, train_set: pd.DataFrame, test_set: pd.DataFrame) -> None:
        """Save the train and test sets into Databricks tables.

        :param train_set: The training DataFrame to be saved.
        :param test_set: The test DataFrame to be saved.
        """
        train_set_with_timestamp = self.spark.createDataFrame(train_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        test_set_with_timestamp = self.spark.createDataFrame(test_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        train_set_with_timestamp.write.mode("overwrite").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.train_set"
        )

        test_set_with_timestamp.write.mode("overwrite").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.test_set"
        )

    def enable_change_data_feed(self) -> None:
        """Enable Change Data Feed for train and test set tables.

        This method alters the tables to enable Change Data Feed functionality.
        """
        self.spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.train_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )

        self.spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.test_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )
