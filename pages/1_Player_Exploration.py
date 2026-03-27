import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set page configuration which gives the browser tab a title and icon, and sets the layout to wide.
st.set_page_config(page_title="Season Stats", page_icon="📈", layout="wide")

# Load data in a cached function for performance
@st.cache_data
def load_data():
    df = pd.read_csv("data/ea_cfb26_players_with_headshots.csv")
    # Extract raw URLs from HTML strings like <img src="https://..." width="60">
    if "Headshot URL" in df.columns:
        df["Headshot URL"] = df["Headshot URL"].astype(str).str.extract(r'src="([^"]+)"', expand=False).fillna(df["Headshot URL"])
    return df


meta_df = load_data()

st.header("Player Overview")

st.write("Search a Player From the Box Below to See Their Ratings and Combine Stats")
player = st.selectbox("Select a player", options=meta_df["Name"])
selected = meta_df[meta_df["Name"] == player].iloc[0]

c1, c2, c3 = st.columns([1, 2, 3])

with c1:
    st.image(selected["Headshot URL"], width=200)
    st.markdown(f"**{selected['Name']}**  \n{selected['Position']} • {selected['Team']}")
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; gap: 4px; margin-top: 0.5rem;">
            <img src="{selected['Team Logo URL']}" style="width: 60px;">
            <img src="{selected['Helmet URL']}" style="width: 60px;">
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.metric("Overall Rating", selected["Overall Rating"])
    st.metric("Speed", selected["Speed"])
    st.metric("Agility", selected["Agility"])
    # etc.

with c3:
    st.metric("Height", selected["Height"])

