import streamlit as st

from mlb_app_utils import GRADE_COLUMNS, load_mlb_data, load_pbr_data


st.set_page_config(page_title="MLB Draft Analysis", layout="wide")

mlb_df = load_mlb_data()
pbr_df = load_pbr_data()

st.markdown("## MLB Draft Analysis")
st.caption("Draft-board comps, PBR athletic metrics, and player-level data views.")

metric_cols = st.columns(4)
metric_cols[0].metric("MLB Draft Players", f"{len(mlb_df):,}")
metric_cols[1].metric("PBR Metric Players", f"{len(pbr_df):,}")
metric_cols[2].metric("Draft Years", f"{int(mlb_df['year'].min())}-{int(mlb_df['year'].max())}")
metric_cols[3].metric("Players With Grades", f"{mlb_df.dropna(subset=GRADE_COLUMNS).shape[0]:,}")

st.markdown("---")

st.markdown("### Start Here")
st.markdown(
        """
        Use the sidebar to move through the app.

        **Similar Player Tool** finds MLB draft comps using size, position, handedness, and scouting grades.

        **PBR Metric Charts** shows the speed, power, hand-speed, and overall metric views from your notebook.

        **Draft Board Data** lets you filter, inspect, and export the combined master CSV.
        """
    )