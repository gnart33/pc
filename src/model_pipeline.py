import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import lightgbm as lgb
import shap
from dataclasses import dataclass


@dataclass
class ModelMetrics:
    """Class to store model evaluation metrics."""

    rmse: float
    mae: float
    directional_accuracy: float
    volatility_capture: float
    extreme_capture: float
    r2_score: float


class ModelPipeline:
    def __init__(self, n_splits: int = 5):
        """Initialize ModelPipeline with number of cross-validation splits."""
        self.n_splits = n_splits
        self.model = None
        self.feature_importance = None
        self.shap_values = None
        self.cv_results = []

    def prepare_train_test_split(
        self, df: pd.DataFrame, target_column: str, test_size: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Prepare train-test split respecting time series order."""
        # Sort by index to ensure time series order
        df = df.sort_index()

        # Calculate split point
        split_idx = int(len(df) * (1 - test_size))

        # Split features and target
        features = df.drop(columns=[target_column])
        target = df[target_column]

        # Create train-test split
        X_train = features.iloc[:split_idx]
        X_test = features.iloc[split_idx:]
        y_train = target.iloc[:split_idx]
        y_test = target.iloc[split_idx:]

        return X_train, X_test, y_train, y_test

    def calculate_directional_accuracy(
        self, y_true: pd.Series, y_pred: pd.Series
    ) -> float:
        """Calculate directional accuracy of predictions."""
        true_direction = np.sign(y_true.diff())
        pred_direction = np.sign(y_pred.diff())

        # Calculate accuracy excluding zero changes
        mask = (true_direction != 0) & (pred_direction != 0)
        return np.mean(true_direction[mask] == pred_direction[mask])

    def calculate_volatility_capture(
        self, y_true: pd.Series, y_pred: pd.Series, window: int = 24
    ) -> float:
        """Calculate how well the model captures price volatility."""
        true_volatility = y_true.rolling(window=window).std()
        pred_volatility = y_pred.rolling(window=window).std()

        # Calculate correlation between true and predicted volatility
        return true_volatility.corr(pred_volatility)

    def calculate_extreme_capture(
        self, y_true: pd.Series, y_pred: pd.Series, threshold: float = 0.15
    ) -> float:
        """Calculate accuracy in predicting extreme price movements."""
        # Calculate percentage changes
        true_changes = y_true.pct_change()
        pred_changes = y_pred.pct_change()

        # Identify extreme movements
        true_extreme = np.abs(true_changes) > threshold
        pred_extreme = np.abs(pred_changes) > threshold

        # Calculate accuracy for extreme events
        if true_extreme.sum() == 0:
            return 0.0

        return np.mean(true_extreme == pred_extreme)

    def evaluate_predictions(
        self, y_true: pd.Series, y_pred: pd.Series
    ) -> ModelMetrics:
        """Calculate all evaluation metrics."""
        metrics = ModelMetrics(
            rmse=np.sqrt(mean_squared_error(y_true, y_pred)),
            mae=mean_absolute_error(y_true, y_pred),
            directional_accuracy=self.calculate_directional_accuracy(y_true, y_pred),
            volatility_capture=self.calculate_volatility_capture(y_true, y_pred),
            extreme_capture=self.calculate_extreme_capture(y_true, y_pred),
            r2_score=1
            - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2),
        )

        return metrics

    def train_model(
        self, X_train: pd.DataFrame, y_train: pd.Series, params: Optional[Dict] = None
    ) -> None:
        """Train LightGBM model with given parameters."""
        if params is None:
            params = {
                "objective": "regression",
                "metric": "rmse",
                "boosting_type": "gbdt",
                "num_leaves": 31,
                "learning_rate": 0.05,
                "feature_fraction": 0.9,
            }

        train_data = lgb.Dataset(X_train, label=y_train)
        self.model = lgb.train(params, train_data, num_boost_round=100)

        # Calculate feature importance
        self.feature_importance = pd.Series(
            self.model.feature_importance(), index=X_train.columns
        ).sort_values(ascending=False)

    def calculate_shap_values(self, X: pd.DataFrame) -> None:
        """Calculate SHAP values for feature importance interpretation."""
        explainer = shap.TreeExplainer(self.model)
        self.shap_values = explainer.shap_values(X)

    def cross_validate(
        self, df: pd.DataFrame, target_column: str, params: Optional[Dict] = None
    ) -> List[ModelMetrics]:
        """Perform time series cross-validation."""
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        cv_results = []

        for train_idx, val_idx in tscv.split(df):
            # Split data
            X_train = df.drop(columns=[target_column]).iloc[train_idx]
            y_train = df[target_column].iloc[train_idx]
            X_val = df.drop(columns=[target_column]).iloc[val_idx]
            y_val = df[target_column].iloc[val_idx]

            # Train model
            self.train_model(X_train, y_train, params)

            # Make predictions
            y_pred = self.model.predict(X_val)

            # Evaluate
            metrics = self.evaluate_predictions(
                y_val, pd.Series(y_pred, index=y_val.index)
            )
            cv_results.append(metrics)

        self.cv_results = cv_results
        return cv_results

    def get_confidence_intervals(
        self, X: pd.DataFrame, n_iterations: int = 100, confidence_level: float = 0.95
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate confidence intervals for predictions using bootstrap."""
        predictions = []

        for _ in range(n_iterations):
            # Random sample with replacement
            sample_idx = np.random.choice(len(X), len(X), replace=True)
            X_sample = X.iloc[sample_idx]

            # Make predictions
            pred = self.model.predict(X_sample)
            predictions.append(pred)

        # Calculate statistics
        predictions = np.array(predictions)
        mean_pred = pd.Series(np.mean(predictions, axis=0), index=X.index)
        lower = pd.Series(
            np.percentile(predictions, (1 - confidence_level) * 100 / 2, axis=0),
            index=X.index,
        )
        upper = pd.Series(
            np.percentile(predictions, 100 - (1 - confidence_level) * 100 / 2, axis=0),
            index=X.index,
        )

        return mean_pred, lower, upper


if __name__ == "__main__":
    # Example usage
    from data_loader import DataLoader
    from feature_engineering import FeatureEngineer

    # Load and prepare data
    loader = DataLoader()
    fe = FeatureEngineer()

    # Load datasets
    prices_df = loader.load_dataset("Day-ahead prices")
    consumption_df = loader.load_dataset("Actual consumption")
    generation_df = loader.load_dataset("Actual generation")
    flows_df = loader.load_dataset("Cross-border physical flows")

    # Prepare features
    target_column = "Germany/Luxembourg [€/MWh] Original resolutions"
    features_df = fe.prepare_features(
        prices_df, consumption_df, generation_df, flows_df, target_column
    )

    # Initialize and run pipeline
    pipeline = ModelPipeline(n_splits=5)
    X_train, X_test, y_train, y_test = pipeline.prepare_train_test_split(
        features_df, target_column
    )

    # Train model
    pipeline.train_model(X_train, y_train)

    # Make predictions
    y_pred = pipeline.model.predict(X_test)

    # Evaluate
    metrics = pipeline.evaluate_predictions(
        y_test, pd.Series(y_pred, index=y_test.index)
    )
    print("\nTest Set Metrics:")
    for metric_name, value in metrics.__dict__.items():
        print(f"{metric_name}: {value:.4f}")

    # Calculate confidence intervals
    mean_pred, lower, upper = pipeline.get_confidence_intervals(X_test)
    print("\nConfidence Intervals Sample:")
    print(
        pd.DataFrame(
            {"mean": mean_pred.head(), "lower": lower.head(), "upper": upper.head()}
        )
    )
