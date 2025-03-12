import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import json

# Set page config
st.set_page_config(page_title="PowerCast Data Explorer", page_icon="⚡", layout="wide")

# Constants
DATADIR = Path("data")
DATE_COLS = ["Start date", "End date"]


def load_metadata():
    """Load metadata from json file"""
    try:
        with open(DATADIR / "metadata.json") as f:
            return json.load(f)
    except:
        return {}


def clean_numeric_columns(df):
    """Clean numeric columns by removing commas and converting to float"""
    for col in df.columns:
        if col not in DATE_COLS:  # Skip date columns
            try:
                # Remove commas and convert to float
                df[col] = df[col].replace({",": ""}, regex=True).astype(float)
            except:
                # If conversion fails, keep the column as is
                continue
    return df


def load_data(file_path):
    """Load CSV data with proper date parsing"""
    # First load with dates
    df = pd.read_csv(
        file_path,
        delimiter=";",
        parse_dates=DATE_COLS,
        date_parser=lambda x: pd.to_datetime(x, format="%b %d, %Y %I:%M %p"),
        na_values=["-"],
    )

    # Clean numeric columns
    df = clean_numeric_columns(df)
    return df


def get_file_list():
    """Get list of available CSV files"""
    csv_files = list(DATADIR.glob("*.csv"))
    return [f.stem for f in csv_files]


def clean_file_name(file_name):
    """Clean file name for display"""
    return file_name.replace("_202301010000_202503050000", "").replace("_", " ")


def display_dataset_overview(df):
    """Display basic dataset information"""
    st.header("Dataset Overview")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Number of Rows", df.shape[0])
    with col2:
        st.metric("Number of Columns", df.shape[1])
    with col3:
        st.metric(
            "Date Range",
            f"{df['Start date'].min().date()} to {df['Start date'].max().date()}",
        )


def display_data_preview(df):
    """Display data preview"""
    st.subheader("Data Preview")
    st.dataframe(df.head(), use_container_width=True)


def display_basic_statistics(df):
    """Display basic statistics for numeric columns"""
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    if len(numeric_cols) > 0:
        st.subheader("Basic Statistics")
        st.dataframe(df[numeric_cols].describe(), use_container_width=True)
    else:
        st.warning("No numeric columns found in the dataset for statistical analysis.")


def create_time_series_plot(df_plot, time_unit):
    """Create time series plot"""
    fig = px.line(
        df_plot,
        title=f"Time series plot ({time_unit.lower() if time_unit != 'Raw data' else 'original'} data)",
        labels={"value": "Value", "Start date": "Date"},
        height=600,
    )

    fig.update_layout(
        showlegend=True,
        legend_title_text="Variables",
        xaxis_title="Date",
        yaxis_title="Value",
        hovermode="x unified",
    )

    return fig


def create_correlation_matrix(df, selected_cols):
    """Create correlation matrix plot"""
    corr = df[selected_cols].corr()
    fig = px.imshow(
        corr,
        title="Correlation Matrix",
        labels=dict(color="Correlation"),
        color_continuous_scale="RdBu",
        aspect="auto",
    )
    return fig


def process_time_series_data(df, selected_cols, time_unit):
    """Process time series data based on selected aggregation"""
    if time_unit != "Raw data":
        agg_map = {
            "Hourly": "H",
            "Daily": "D",
            "Weekly": "W",
            "Monthly": "M",
        }
        return (
            df.set_index("Start date")[selected_cols]
            .resample(agg_map[time_unit])
            .mean()
        )
    return df.set_index("Start date")[selected_cols]


def display_visualizations(df):
    """Handle data visualization section"""
    st.header("Data Visualization")

    numeric_cols = [
        col
        for col in df.columns
        if col not in DATE_COLS and df[col].dtype in ["float64", "int64"]
    ]

    if len(numeric_cols) > 0:
        selected_cols = st.multiselect(
            "Select columns to plot",
            numeric_cols,
            default=[numeric_cols[0]] if numeric_cols else None,
        )

        if selected_cols:
            time_unit = st.selectbox(
                "Select time aggregation",
                ["Raw data", "Hourly", "Daily", "Weekly", "Monthly"],
            )

            # Process data
            df_plot = process_time_series_data(df, selected_cols, time_unit)

            # Display time series plot
            fig = create_time_series_plot(df_plot, time_unit)
            st.plotly_chart(fig, use_container_width=True)

            # Display correlation matrix if multiple columns selected
            if len(selected_cols) > 1:
                st.subheader("Correlation Matrix")
                fig_corr = create_correlation_matrix(df, selected_cols)
                st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.warning("No numeric columns available for plotting.")


def main():
    """Main application function"""
    st.title("⚡ PowerCast Data Explorer")
    st.write("Explore power generation and consumption data from various sources")

    # Sidebar
    st.sidebar.header("Data Selection")

    # File selection
    file_names = sorted(get_file_list())
    selected_file = st.sidebar.selectbox(
        "Choose a dataset",
        file_names,
        format_func=clean_file_name,
    )

    if selected_file:
        try:
            # Load and process data
            file_path = DATADIR / f"{selected_file}.csv"
            df = load_data(file_path)

            # Display sections
            display_dataset_overview(df)
            display_data_preview(df)
            display_basic_statistics(df)
            display_visualizations(df)

        except Exception as e:
            st.error(f"Error loading or processing the data: {str(e)}")


if __name__ == "__main__":
    main()
