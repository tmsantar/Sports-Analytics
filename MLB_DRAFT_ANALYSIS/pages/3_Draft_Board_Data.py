import streamlit as st

from mlb_app_utils import GRADE_COLUMNS, load_mlb_data, load_pbr_data


st.set_page_config(page_title="Draft Board Data", layout="wide")

mlb_df = load_mlb_data()
pbr_df = load_pbr_data()

st.markdown("## Draft Board Data")
st.caption("Filter the MLB draft master CSV and PBR metric file.")

dataset = st.radio("Dataset", ["MLB Draft Master", "PBR Metrics"], horizontal=True)

if dataset == "MLB Draft Master":
    df = mlb_df.copy()
    years = sorted(df["year"].dropna().astype(int).unique())
    positions = sorted(df["position"].dropna().astype(str).unique())

    with st.sidebar:
        st.markdown("### MLB Filters")
        selected_years = st.multiselect("Years", years, default=years)
        selected_positions = st.multiselect("Positions", positions)
        search = st.text_input("Player search")
        grades_only = st.toggle("Require all hitter grades", value=False)

    if selected_years:
        df = df[df["year"].isin(selected_years)]
    if selected_positions:
        df = df[df["position"].isin(selected_positions)]
    if search:
        df = df[df["player_name"].astype(str).str.contains(search, case=False, na=False)]
    if grades_only:
        df = df.dropna(subset=GRADE_COLUMNS)

    show_cols = [
        "year",
        "rank",
        "player_name",
        "position",
        "school",
        "height",
        "weight",
        "bats",
        "throws",
        *GRADE_COLUMNS,
        "profile_url",
    ]

else:
    df = pbr_df.copy()
    positions = sorted(df["article_position"].dropna().astype(str).unique())

    with st.sidebar:
        st.markdown("### PBR Filters")
        selected_positions = st.multiselect("Positions", positions)
        search = st.text_input("Player search")
        min_ev = st.slider(
            "Minimum max EV",
            0.0,
            float(df["max_exit_velocity"].max()),
            0.0,
            step=0.5,
        )

    if selected_positions:
        df = df[df["article_position"].isin(selected_positions)]
    if search:
        df = df[df["player_name"].astype(str).str.contains(search, case=False, na=False)]
    df = df[df["max_exit_velocity"].fillna(0) >= min_ev]

    show_cols = [
        "rank",
        "player_name",
        "article_position",
        "article_school",
        "height",
        "weight",
        "sixty_time",
        "speed_score",
        "max_exit_velocity",
        "max_hand_speed",
        "max_bat_speed",
        "bats",
        "throws",
        "profile_url",
    ]

st.metric("Rows", f"{len(df):,}")
st.dataframe(df[[col for col in show_cols if col in df.columns]], use_container_width=True, hide_index=True)

csv = df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download filtered data",
    data=csv,
    file_name=f"{dataset.lower().replace(' ', '_')}.csv",
    mime="text/csv",
)
