import streamlit as st
import pandas as pd
from pathlib import Path
import json

# Set page config
st.set_page_config(page_title="PowerCast Data Explorer", page_icon="⚡", layout="wide")

# Constants
DATADIR = Path("data")
DATE_COLS = ["Start date", "End date"]


def main():
    st.title("⚡ PowerCast Data Explorer")


if __name__ == "__main__":
    main()
