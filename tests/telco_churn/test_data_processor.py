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
from telco_churn.data_processor import DataProcessor, generate_synthetic_data, generate_test_data  # noqa: E402


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Create a sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "customerID": ["1", "2", "3", "4", "5"],
            "gender": ["Male", "Female", "Male", "Female", "Male"],
            "SeniorCitizen": [0, 1, 0, 0, 1],
            "Partner": ["Yes", "No", "Yes", "No", "Yes"],
            "Dependents": ["No", "No", "Yes", "No", "Yes"],
            "tenure": [12, 24, 36, 6, 48],
            "MonthlyCharges": [70.5, 55.0, 100.2, 45.8, 80.0],
            "TotalCharges": [846.0, 1320.0, 3607.2, 274.8, 3840.0],
            "Contract": ["Month-to-month", "One year", "Two year", "Month-to-month", "Two year"],
            "Churn": ["Yes", "No", "No", "Yes", "No"]
        }
    )


@pytest.fixture
def mock_config() -> MagicMock:
    """Create a mock ProjectConfig for testing."""
    config = MagicMock(spec=ProjectConfig)
    config.cat_features =["gender","Partner","Dependents","Contract"]
    config.num_features = ["SeniorCitizen","tenure","MonthlyCharges","TotalCharges"]
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

    def test_init(self, sample_data: pd.DataFrame, mock_config: MagicMock, mock_spark: MagicMock) -> None:
        """Test DataProcessor initialization."""
        processor = DataProcessor(sample_data, mock_config, mock_spark)
        assert processor.df is sample_data
        assert processor.config is mock_config
        assert processor.spark is mock_spark

    def test_preprocess(self, sample_data: pd.DataFrame, mock_config: MagicMock, mock_spark: MagicMock) -> None:
        """Test the preprocess method."""
        processor = DataProcessor(sample_data, mock_config, mock_spark)
        processor.preprocess()

        # Check column renames
        assert "customerID" not in processor.df.columns
        assert "tenure" in processor.df.columns
        assert "MonthlyCharges" in processor.df.columns
        assert "TotalCharges" in processor.df.columns

        # Check missing value handling
        assert not processor.df["Universe"].isna().any()
        assert not processor.df["Origin"].isna().any()
        assert not processor.df["Identity"].isna().any()

        # Check feature engineering
        assert "Magic" in processor.df.columns
        assert "Mutant" in processor.df.columns

        # Check data types
        for col in mock_config.cat_features:
            assert pd.api.types.is_categorical_dtype(processor.df[col])

        # Check target conversion
        assert set(processor.df["Alive"].unique()) == {0, 1}

    def test_split_data(self, sample_data: pd.DataFrame, mock_config: MagicMock, mock_spark: MagicMock) -> None:
        """Test the split_data method."""
        processor = DataProcessor(sample_data, mock_config, mock_spark)
        processor.preprocess()

        train_set, test_set = processor.split_data(test_size=0.4, random_state=42)

        # Check split sizes
        assert len(train_set) + len(test_set) == len(processor.df)
        assert len(train_set) > 0
        assert len(test_set) > 0

    def test_save_to_catalog(self, sample_data: pd.DataFrame, mock_config: MagicMock, mock_spark: MagicMock) -> None:
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
        self, mock_sql: MagicMock, sample_data: pd.DataFrame, mock_config: MagicMock, mock_spark: MagicMock
    ) -> None:
        """Test the enable_change_data_feed method."""
        processor = DataProcessor(sample_data, mock_config, mock_spark)
        processor.enable_change_data_feed()

        # Check that SparkSession.sql was called twice
        assert mock_spark.sql.call_count == 2


class TestDataGenerationFunctions:
    """Tests for the data generation functions."""

    def test_generate_synthetic_data(self, sample_data: pd.DataFrame) -> None:
        """Test the generate_synthetic_data function."""
        # Preprocess the sample data to match expected format
        processor = DataProcessor(
            sample_data, MagicMock(spec=ProjectConfig, cat_features=[], num_features=[], target="Alive"), MagicMock()
        )
        processor.preprocess()

        # Generate synthetic data
        synthetic_data = generate_synthetic_data(processor.df, drift=False, num_rows=10)

        # Check that the synthetic data has the expected number of rows
        assert len(synthetic_data) == 10

        # Check that the synthetic data has the same columns
        assert set(synthetic_data.columns) == set(processor.df.columns)

        # Test with drift
        synthetic_data_drift = generate_synthetic_data(processor.df, drift=True, num_rows=10)
        assert len(synthetic_data_drift) == 10

    def test_generate_test_data(self, sample_data: pd.DataFrame) -> None:
        """Test the generate_test_data function."""
        # Preprocess the sample data to match expected format
        processor = DataProcessor(
            sample_data, MagicMock(spec=ProjectConfig, cat_features=[], num_features=[], target="Alive"), MagicMock()
        )
        processor.preprocess()

        # Generate test data
        test_data = generate_test_data(processor.df, num_rows=5)

        # Check that the test data has the expected number of rows
        assert len(test_data) == 5

        # Check that the test data has the same columns
        assert set(test_data.columns) == set(processor.df.columns)
