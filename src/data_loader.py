import json
import pandas as pd
from pathlib import Path
from typing import Dict, Optional

DATE_COLS = ["Start date", "End date"]


class DataLoader:
    def __init__(self, data_dir: str = "data/raw"):
        """Initialize DataLoader with data directory path."""
        self.data_dir = Path(data_dir)
        self.metadata = self._load_metadata()
        self.datasets: Dict[str, Optional[pd.DataFrame]] = {}

    def _load_metadata(self) -> dict:
        """Load metadata from JSON file."""
        if not (self.data_dir / "metadata.json").exists():
            self._create_metadata(self.data_dir / "metadata.json")
        with open(self.data_dir / "metadata.json", "r") as f:
            return json.load(f)

    def _create_metadata(self, metadata_path: str) -> None:
        """Create metadata file if it doesn't exist."""
        metadata = {}
        for f in self.data_dir.glob("*.csv"):
            filename = f.stem.split("_202301010000_202503050000")[0].replace("_", " ")
            df = pd.read_csv(f, delimiter=";", low_memory=False)
            metadata[filename] = {
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "file": str(f.stem),
            }
        sorted_metadata = dict(sorted(metadata.items()))

        with open(metadata_path, "w") as f:
            json.dump(sorted_metadata, f)

    def list_available_datasets(self) -> list:
        """List all available datasets."""
        return list(self.metadata.keys())

    def load_dataset(self, dataset_name: str) -> pd.DataFrame:
        """Load a specific dataset by name."""
        if dataset_name not in self.metadata:
            raise ValueError(f"Dataset {dataset_name} not found in metadata")

        if dataset_name not in self.datasets:
            file_name = self.metadata[dataset_name]["file"]
            file_path = self.data_dir / f"{file_name}.csv"

            # Read the CSV file
            df = pd.read_csv(file_path, delimiter=";", na_values=["-"])

            # Convert date columns to datetime
            df["Start date"] = pd.to_datetime(
                df["Start date"], format="%b %d, %Y %I:%M %p"
            )
            df["End date"] = pd.to_datetime(df["End date"], format="%b %d, %Y %I:%M %p")

            # Set Start date as index
            # df.set_index("Start date", inplace=True)

            self.datasets[dataset_name] = df

        return self.datasets[dataset_name]

    def get_data_summary(self, dataset_name: str) -> dict:
        """Get summary statistics for a dataset."""
        df = self.load_dataset(dataset_name)
        return {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "missing_values": df.isnull().sum().to_dict(),
            # "memory_usage": df.memory_usage(deep=True).sum() / 1024**2,  # MB
            "date_range": (df.index.min(), df.index.max()),
        }

    def get_dataset_info(self, dataset_name: str) -> dict:
        """Get metadata information for a specific dataset."""
        if dataset_name not in self.metadata:
            raise ValueError(f"Dataset {dataset_name} not found in metadata")
        return self.metadata[dataset_name]
