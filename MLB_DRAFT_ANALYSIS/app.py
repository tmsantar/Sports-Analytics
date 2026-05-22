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

left, right = st.columns([1.1, 1])
with left:
    st.markdown("### Start Here")
    st.markdown(
        """
        Use the sidebar to move through the app.

        **Similar Player Tool** finds MLB draft comps using size, position, handedness, and scouting grades.

        **PBR Metric Charts** shows the speed, power, hand-speed, and overall metric views from your notebook.

        **Draft Board Data** lets you filter, inspect, and export the combined master CSV.
        """
    )

with right:
    st.markdown("### Dominic Santarelli Snapshot")
    dom = mlb_df[mlb_df["player_name"].astype(str).str.contains("Dominic Santarelli", case=False, na=False)]
    if dom.empty:
        st.warning("Dominic Santarelli was not found in the MLB master CSV. Regenerate the scraper output first.")
    else:
        row = dom.iloc[0]
        st.markdown(f"#### {row['player_name']} | {row['position']}")
        st.markdown(f"Rank {row['rank']} | {row['height']} | {row['weight']} lbs | B/T {row['bats']}/{row['throws']}")
        st.markdown(
            f"Grades: Hit {row['hit']} | Power {row['power']} | Run {row['run']} | "
            f"Arm {row['arm']} | Field {row['field']} | Overall {row['overall']}"
        )
        image_url = row.get("profile_image_url")
        if image_url:
            st.image(image_url, width=240)
