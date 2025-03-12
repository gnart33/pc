import json
from pathlib import Path
import pandas as pd


def metadata(datadir):
    metadata = {}
    for f in datadir.glob("*.csv"):
        filename = f.stem.split("_202301010000_202503050000")[0].replace("_", " ")
        df = pd.read_csv(f, delimiter=";", low_memory=False)
        metadata[filename] = {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "file": str(f.stem),
        }
    sorted_metadata = dict(sorted(metadata.items()))

    with open("data/metadata.json", "w") as f:
        json.dump(sorted_metadata, f)


if __name__ == "__main__":
    datadir = Path(__file__).parent / "data"
    metadata(datadir)
