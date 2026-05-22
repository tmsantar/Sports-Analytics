import streamlit as st

from mlb_app_utils import find_pbr_player, load_mlb_data, load_pbr_data, player_options, render_player_card


st.set_page_config(page_title="Player Reporter", layout="wide")

mlb_df = load_mlb_data()
pbr_df = load_pbr_data()

st.markdown("## Player Reporter")
st.caption("Search for a player and read their scouting profile and PBR details.")

all_names = sorted(
    set(mlb_df["player_name"].dropna().astype(str).unique())
    | set(pbr_df["player_name"].dropna().astype(str).unique())
)

selected_player_name = st.selectbox("Select player", options=all_names)

selected_player = mlb_df[mlb_df["player_name"].eq(selected_player_name)]
if selected_player.empty:
    st.warning("No MLB draft record found for this player. Showing PBR-only details if available.")
    selected_player = None
else:
    selected_player = selected_player.iloc[0]

selected_pbr = find_pbr_player(pbr_df, selected_player_name)

if selected_player is None and selected_pbr is None:
    st.error("No data was found for this player.")
    st.stop()

if selected_player is not None:
    render_player_card(selected_player, "Player Reporter", selected_pbr)
else:
    st.markdown(f"### {selected_player_name}")
    st.info("No MLB draft master record available.")
    if selected_pbr is not None:
        st.markdown(f"##### PBR Rank: {selected_pbr.get('rank', 'N/A')} | Position: {selected_pbr.get('article_position', 'N/A')}")
        st.markdown(f"##### School: {selected_pbr.get('article_school', 'N/A')}")
        st.markdown(f"##### Size: {selected_pbr.get('height', 'N/A')} | {selected_pbr.get('weight', 'N/A')} lbs")
        st.markdown(f"##### B/T: {selected_pbr.get('bats', 'N/A')}/{selected_pbr.get('throws', 'N/A')}")

if selected_pbr is not None and selected_player is None:
    st.markdown("---")
    st.subheader("PBR Metrics")
    st.dataframe(
        selected_pbr[
            [
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
            if "rank" in selected_pbr.index
            else []
        ]
        .to_frame()
        .T,
        use_container_width=True,
        hide_index=True,
    )
