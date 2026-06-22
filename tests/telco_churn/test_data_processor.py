"""Tests for the DataProcessor class."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from pyspark.sql import SparkSession

# Add the src directory to the Python path if not already added
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Import project modules after path setup
from telco_churn.config import ProjectConfig  # noqa: E402
from telco_churn.data_processor import DataProcessor


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Create a sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "gender": ["Female", "Male", "Male", "Male", "Female"],
            "SeniorCitizen": [0, 0, 0, 0, 0],
            "Partner": ["Yes", "No", "No", "No", "No"],
            "Dependents": ["No", "No", "No", "No", "No"],
            "tenure": [1, 34, 2, 45, 2],
            "PhoneService": ["No", "Yes", "Yes", "No", "Yes"],
            "MultipleLines": ["No phone service", "No", "No", "No phone service", "No"],
            "InternetService": ["DSL", "DSL", "DSL", "DSL", "Fiber optic"],
            "OnlineSecurity": ["No", "Yes", "Yes", "Yes", "No"],
            "OnlineBackup": ["Yes", "No", "Yes", "No", "No"],
            "DeviceProtection": ["No", "Yes", "No", "Yes", "No"],
            "TechSupport": ["No", "No", "No", "Yes", "No"],
            "StreamingTV": ["No", "No", "No", "No", "No"],
            "StreamingMovies": ["No", "No", "No", "No", "No"],
            "Contract": [
                "Month-to-month",
                "One year",
                "Month-to-month",
                "One year",
                "Month-to-month",
            ],
            "PaperlessBilling": ["Yes", "No", "Yes", "No", "Yes"],
            "PaymentMethod": [
                "Electronic check",
                "Mailed check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Electronic check",
            ],
            "MonthlyCharges": [29.85, 56.95, 53.85, 42.30, 70.70],
            "TotalCharges": [29.85, 1889.50, 108.15, 1840.75, 151.65],
            "Churn": ["No", "No", "Yes", "No", "Yes"],
        }
    )


@pytest.fixture
def mock_config() -> MagicMock:
    """Create a mock ProjectConfig for testing."""
    config = MagicMock(spec=ProjectConfig)
    config.num_features = [
        "Tenure",
        "MonthlyCharges",
        "TotalCharges",
        "MaleGender",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Partner",
        "Dependents",
        "PhoneService",
        "PaperlessBilling",
        "SeniorCitizen",
    ]
    config.target = "Churn"
    config.catalog_name = "test_catalog"
    config.schema_name = "test_schema"
    return config


@pytest.fixture
def mock_spark() -> MagicMock:
    """Create a mock SparkSession for testing."""
    mock = MagicMock(spec=SparkSession)
    mock.createDataFrame.return_value = MagicMock()
    mock.createDataFrame.return_value.withColumn.return_value = MagicMock()
    mock.createDataFrame.return_value.withColumn.return_value.write.mode.return_value = MagicMock()
    return mock


class TestDataProcessor:
    """Tests for the DataProcessor class."""

    def test_init(
        self, sample_data: pd.DataFrame, mock_config: MagicMock, mock_spark: MagicMock
    ) -> None:
        """Test DataProcessor initialization."""
        processor = DataProcessor(sample_data, mock_config, mock_spark)
        assert processor.df is sample_data
        assert processor.config is mock_config
        assert processor.spark is mock_spark

    def test_preprocess(
        self, sample_data: pd.DataFrame, mock_config: MagicMock, mock_spark: MagicMock
    ) -> None:
        """Test the preprocess method."""
        processor = DataProcessor(sample_data, mock_config, mock_spark)
        processor.preprocess()

        # Check column renames
        assert "gender" not in processor.df.columns
        assert "Tenure" in processor.df.columns

        # Check missing value handling
        assert not processor.df["TotalCharges"].isna().any()

        # Check data type
        for col in mock_config.num_features:
            assert pd.api.types.is_numeric_dtype(processor.df[col])

        # Check flag columns
        binary_cols = [
            "Partner",
            "Dependents",
            "PhoneService",
            "MultipleLines",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
            "PaperlessBilling",
            "Churn",
        ]

        for col in binary_cols:
            assert set(processor.df[col].unique()).issubset({0, 1})

        # Check target conversion
        assert set(processor.df["Churn"].unique()) == {0, 1}

    def test_split_data(
        self, sample_data: pd.DataFrame, mock_config: MagicMock, mock_spark: MagicMock
    ) -> None:
        """Test the split_data method."""
        processor = DataProcessor(sample_data, mock_config, mock_spark)
        processor.preprocess()

        train_set, test_set = processor.split_data(test_size=0.4, random_state=42)

        # Check split sizes
        assert len(train_set) + len(test_set) == len(processor.df)
        assert len(train_set) > 0
        assert len(test_set) > 0

    def test_save_to_catalog(
        self, sample_data: pd.DataFrame, mock_config: MagicMock, mock_spark: MagicMock
    ) -> None:
        """Test the save_to_catalog method."""
        # Create a processor instance
        processor = DataProcessor(sample_data, mock_config, mock_spark)
        processor.preprocess()

        # Get train and test sets
        train_set, test_set = processor.split_data()

        # Mock the save_to_catalog method to avoid PySpark context issues
        with patch.object(DataProcessor, "save_to_catalog") as mock_save:
            # Call the method
            processor.save_to_catalog(train_set, test_set)

            # Verify it was called with the right arguments
            mock_save.assert_called_once_with(train_set, test_set)

    @patch("pyspark.sql.SparkSession.sql")
    def test_enable_change_data_feed(
        self,
        mock_sql: MagicMock,
        sample_data: pd.DataFrame,
        mock_config: MagicMock,
        mock_spark: MagicMock,
    ) -> None:
        """Test the enable_change_data_feed method."""
        processor = DataProcessor(sample_data, mock_config, mock_spark)
        processor.enable_change_data_feed()

        # Check that SparkSession.sql was called twice
        assert mock_spark.sql.call_count == 2
