import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from sklearn.preprocessing import StandardScaler


class FeatureEngineer:
    def __init__(self):
        """Initialize FeatureEngineer with necessary scalers and transformers."""
        self.scalers: Dict[str, StandardScaler] = {}
        self.feature_names: List[str] = []

    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features from datetime index."""
        df = df.copy()

        # Basic time features
        df["hour"] = df.index.hour
        df["day"] = df.index.day
        df["weekday"] = df.index.weekday
        df["month"] = df.index.month
        df["quarter"] = df.index.quarter

        # Cyclical encoding of time features
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["day_sin"] = np.sin(2 * np.pi * df["day"] / 31)
        df["day_cos"] = np.cos(2 * np.pi * df["day"] / 31)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        return df

    def create_lag_features(
        self, df: pd.DataFrame, columns: List[str], lags: List[int]
    ) -> pd.DataFrame:
        """Create lagged features for specified columns."""
        df = df.copy()

        for col in columns:
            for lag in lags:
                df[f"{col}_lag_{lag}"] = df[col].shift(lag)

        return df

    def create_rolling_features(
        self, df: pd.DataFrame, columns: List[str], windows: List[int]
    ) -> pd.DataFrame:
        """Create rolling statistics features."""
        df = df.copy()

        for col in columns:
            for window in windows:
                # Rolling mean
                df[f"{col}_rolling_mean_{window}"] = (
                    df[col].rolling(window=window).mean()
                )
                # Rolling std
                df[f"{col}_rolling_std_{window}"] = df[col].rolling(window=window).std()
                # Rolling min/max
                df[f"{col}_rolling_min_{window}"] = df[col].rolling(window=window).min()
                df[f"{col}_rolling_max_{window}"] = df[col].rolling(window=window).max()

        return df

    def create_price_volatility_features(
        self, df: pd.DataFrame, price_column: str, windows: List[int]
    ) -> pd.DataFrame:
        """Create price volatility features."""
        df = df.copy()

        for window in windows:
            # Rolling volatility
            df[f"price_volatility_{window}"] = (
                df[price_column].rolling(window=window).std()
            )
            # Rolling price range
            df[f"price_range_{window}"] = (
                df[price_column].rolling(window=window).max()
                - df[price_column].rolling(window=window).min()
            )
            # Rolling price momentum
            df[f"price_momentum_{window}"] = (
                df[price_column] - df[price_column].rolling(window=window).mean()
            )

        return df

    def create_cross_border_features(self, flows_df: pd.DataFrame) -> pd.DataFrame:
        """Create features from cross-border flows."""
        df = flows_df.copy()

        # Calculate net flows
        export_cols = [col for col in df.columns if "(export)" in col]
        import_cols = [col for col in df.columns if "(import)" in col]

        df["total_exports"] = df[export_cols].sum(axis=1)
        df["total_imports"] = df[import_cols].sum(axis=1)
        df["net_flow"] = df["total_exports"] - df["total_imports"]

        # Calculate flow volatility
        df["flow_volatility"] = df["net_flow"].rolling(window=24).std()

        return df

    def create_generation_mix_features(self, gen_df: pd.DataFrame) -> pd.DataFrame:
        """Create features from generation mix."""
        df = gen_df.copy()

        # Group by generation type
        renewable_cols = [
            "Wind offshore",
            "Wind onshore",
            "Photovoltaics",
            "Hydropower",
            "Biomass",
            "Other renewable",
        ]
        conventional_cols = [
            "Nuclear",
            "Lignite",
            "Hard coal",
            "Fossil gas",
            "Other conventional",
        ]

        # Calculate total generation by type
        for col_type in [renewable_cols, conventional_cols]:
            cols = [
                col
                for col in df.columns
                if any(type_name in col for type_name in col_type)
            ]
            type_name = "renewable" if col_type == renewable_cols else "conventional"
            df[f"total_{type_name}"] = df[cols].sum(axis=1)

        # Calculate renewable percentage
        df["renewable_percentage"] = df["total_renewable"] / (
            df["total_renewable"] + df["total_conventional"]
        )

        return df

    def scale_features(
        self, df: pd.DataFrame, columns: List[str], fit: bool = True
    ) -> pd.DataFrame:
        """Scale specified features using StandardScaler."""
        df = df.copy()

        for col in columns:
            if fit:
                self.scalers[col] = StandardScaler()
                df[col] = self.scalers[col].fit_transform(df[[col]])
            else:
                if col in self.scalers:
                    df[col] = self.scalers[col].transform(df[[col]])
                else:
                    raise ValueError(
                        f"Scaler for column {col} not found. Need to fit first."
                    )

        return df

    def prepare_features(
        self,
        prices_df: pd.DataFrame,
        consumption_df: pd.DataFrame,
        generation_df: pd.DataFrame,
        flows_df: pd.DataFrame,
        target_column: str,
    ) -> pd.DataFrame:
        """Prepare all features for modeling."""
        # Create base dataframe with prices
        df = prices_df[[target_column]].copy()

        # Add time features
        df = self.create_time_features(df)

        # Add consumption features
        consumption_cols = [col for col in consumption_df.columns if "[MWh]" in col]
        df = df.join(consumption_df[consumption_cols])

        # Add generation features
        generation_df = self.create_generation_mix_features(generation_df)
        df = df.join(generation_df)

        # Add flow features
        flows_df = self.create_cross_border_features(flows_df)
        df = df.join(flows_df)

        # Create lag features for key variables
        key_cols = [target_column, "total_renewable", "total_conventional", "net_flow"]
        df = self.create_lag_features(
            df, key_cols, lags=[1, 24, 48, 168]
        )  # 1h, 24h, 48h, 1 week

        # Create rolling features
        df = self.create_rolling_features(
            df, key_cols, windows=[24, 168]
        )  # 24h, 1 week

        # Create price volatility features
        df = self.create_price_volatility_features(df, target_column, windows=[24, 168])

        # Store feature names
        self.feature_names = [col for col in df.columns if col != target_column]

        return df


if __name__ == "__main__":
    # Example usage
    from data_loader import DataLoader

    # Load data
    loader = DataLoader()
    prices_df = loader.load_dataset("Day-ahead prices")
    consumption_df = loader.load_dataset("Actual consumption")
    generation_df = loader.load_dataset("Actual generation")
    flows_df = loader.load_dataset("Cross-border physical flows")

    # Initialize feature engineer
    fe = FeatureEngineer()

    # Prepare features
    target_column = "Germany/Luxembourg [€/MWh] Original resolutions"
    features_df = fe.prepare_features(
        prices_df, consumption_df, generation_df, flows_df, target_column
    )

    print("Feature Engineering Complete")
    print(f"Number of features created: {len(fe.feature_names)}")
    print("\nSample features:")
    print(features_df.head())
