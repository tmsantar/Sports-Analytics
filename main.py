import pandas as pd
import streamlit as st
import numpy as np

#Set page configuration which gives the browser tab a title and icon, and sets the layout to wide.
st.set_page_config( page_title="NCAA Player Comparison", page_icon="🏈", layout="wide")

# Load dcd ata in a cached function for performance
@st.cache_data
def load_data():
    return pd.read_csv("data/ea_cfb26_players_with_headshots.csv")


meta_df = load_data()

# Sidebar content
with st.sidebar:   

    # About me portion of the sidebar with links to LinkedIn and GitHub
    st.markdown("Built by **Tommy Santarelli**")
    st.caption("Business Analytics Major at Notre Dame")
    st.markdown("🔗 [LinkedIn](https://www.linkedin.com/in/tommy-santarelli-792651329/)")
    st.markdown("🐙 [GitHub](https://github.com/tmsantar)")
    st.markdown("---")

    # Description of where the data is sourced from
    st.markdown("The dataset used is sourced from the [NFL Next Gen Stats]"
    "(https://nextgenstats.nfl.com/stats/receiving#yards) for the 2025 regular season.")


st.title("NCAA Player Comparison 🏈")

st.write("""
### About This App

This app allows users to compare the different ratings and combine statistics of NCAA football players.
         The data is taken from the popular video game, EA Sports' NCAA Football 26,
         which provides players ratings based on real-world performance.
        Additionally, combine statistics were added to compare players' ratings with their
         real world physical attributes.""")

st.markdown("---")

# Final note encouraging users to explore the app
st.info("👈 Use the sidebar to navigate between pages and start exploring the data!")