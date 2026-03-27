import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set page configuration which gives the browser tab a title and icon, and sets the layout to wide.
st.set_page_config(page_title="Season Stats", page_icon="📈", layout="wide")

# Load data in a cached function for performance
@st.cache_data
def load_data():
    return pd.read_csv("data/ea_cfb26_players_with_headshots.csv")


meta_df = load_data()

st.header("Player Overview")

st.write("Search a Player From the Box Below to See Their Ratings and Combine Stats")
st.selectbox("Select a Player", options=meta_df["name"].unique())