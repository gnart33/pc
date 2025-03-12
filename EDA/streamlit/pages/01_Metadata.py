import streamlit as st
import pandas as pd
from pathlib import Path
import json

# Set page config
st.set_page_config(
    page_title="PowerCast Metadata Explorer", page_icon="📊", layout="wide"
)

# Constants
DATADIR = Path("data")


def load_metadata():
    """Load metadata from json file"""
    try:
        with open(DATADIR / "metadata.json") as f:
            return json.load(f)
    except:
        return {}


def group_datasets():
    """Group datasets by their type/purpose"""
    return {
        "Generation": [
            "Actual generation",
            "Forecasted generation Day-Ahead",
            "Generation Forecast Intraday",
            "Installed generation capacity",
        ],
        "Consumption": ["Actual consumption", "Forecasted consumption"],
        "Balancing & Reserve": [
            "Automatic Frequency Restoration Reserve",
            "Manual Frequency Restoration Reserve",
            "Frequency Containment Reserve",
            "Balancing energy",
            "Imported balancing services",
            "Exported balancing services",
        ],
        "Market & Exchange": [
            "Day-ahead prices",
            "Cross-border physical flows",
            "Scheduled commercial exchanges",
        ],
        "Costs": ["Costs of TSOs  without costs of DSOs"],
    }


def display_dataset_info(dataset_name, metadata):
    """Display detailed information about a dataset"""
    st.subheader(f"📄 {dataset_name}")

    if dataset_name in metadata:
        data = metadata[dataset_name]

        # Display file info
        col1, col2 = st.columns(2)
        with col1:
            st.write("**File Information:**")
            st.write(f"- Number of columns: {len(data['columns'])}")
            if "shape" in data:
                st.write(f"- Number of rows: {data['shape'][0]}")

        # Display columns with descriptions
        with col2:
            st.write("**Columns:**")
            for col in data["columns"]:
                st.write(f"- {col}")
    else:
        st.warning(f"No metadata available for {dataset_name}")


def main():
    st.title("📊 PowerCast Metadata Explorer")
    st.write("Explore the structure and content of PowerCast datasets")

    # Load metadata
    metadata = load_metadata()
    dataset_groups = group_datasets()

    # Create tabs for different dataset groups
    tabs = st.tabs(list(dataset_groups.keys()))

    for tab, (group_name, datasets) in zip(tabs, dataset_groups.items()):
        with tab:
            st.header(f"{group_name} Datasets")

            # Create expandable sections for each dataset in the group
            for dataset in datasets:
                with st.expander(f"{dataset} details"):
                    display_dataset_info(dataset, metadata)

            # Add relationships between datasets if they exist
            if group_name == "Generation":
                st.info(
                    """
                **Dataset Relationships:**
                - Actual generation ↔ Forecasted generation (Day-Ahead & Intraday): Compare actual vs predicted values
                - Installed capacity provides context for generation limits
                """
                )
                st.text_input("Enter a dataset name to compare")
            elif group_name == "Consumption":
                st.info(
                    """
                **Dataset Relationships:**
                - Actual consumption ↔ Forecasted consumption: Compare actual vs predicted values
                """
                )
            elif group_name == "Balancing & Reserve":
                st.info(
                    """
                **Dataset Relationships:**
                - Different types of reserves (Frequency Containment, Automatic/Manual Restoration) work together
                - Balancing energy shows the actual energy used for grid stabilization
                """
                )


if __name__ == "__main__":
    main()
