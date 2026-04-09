import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

# Set page configuration which gives the browser tab a title and icon, and sets the layout to wide.
st.set_page_config(page_title="Season Stats", page_icon="📈", layout="wide")


STAT_GROUPS = {
    "General": [
        "Speed",
        "Acceleration",
        "Strength",
        "Agility",
        "Awareness",
        "Jumping",
        "Injury",
        "Stamina",
        "Toughness",
    ],
    "Ballcarrier": [
        "Carrying",
        "Break Tackle",
        "Trucking",
        "Change of Direction",
        "Ball Carrier Vision",
        "Stiff Arm",
        "Spin Move",
        "Juke Move",
        "Break Sack",
    ],
    "Blocking": [
        "Run Block",
        "Pass Block",
        "Impact Blocking",
        "Run Block Power",
        "Run Block Finesse",
        "Pass Block Power",
        "Pass Block Finesse",
        "Lead Block",
    ],
    "Passing": [
        "Throw Power",
        "Throw Under Pressure",
        "Throw Accuracy Short",
        "Throw Accuracy Mid",
        "Throw Accuracy Deep",
        "Throw on the Run",
        "Play Action",
    ],
    "Defense": [
        "Tackle",
        "Power Moves",
        "Finesse Moves",
        "Block Shedding",
        "Pursuit",
        "Play Recognition",
        "Man Coverage",
        "Zone Coverage",
        "Hit Power",
        "Press",
    ],
    "Receiving": [
        "Catching",
        "Spectacular Catch",
        "Catch in Traffic",
        "Short Route Running",
        "Medium Route Running",
        "Deep Route Running",
    ],
    "Kicking": [
        "Kick Power",
        "Kick Accuracy",
        "Kick Return",
    ],
}


def valid_image_url(value):
    if pd.isna(value):
        return None
    value = str(value).strip()
    if not value or value.lower() == "nan":
        return None
    return value


def format_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value or value.lower() == "nan":
            return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def rating_color(value):
    if value is None:
        return "#6b7280"
    if value < 50:
        return "#ef4444"
    if value < 65:
        return "#f97316"
    if value < 80:
        return "#eab308"
    return "#22c55e"


def render_stat_row(label, value):
    numeric_value = format_value(value)
    display_value = "--" if numeric_value is None else str(numeric_value)
    bar_width = 0 if numeric_value is None else max(0, min(100, numeric_value))
    bar_color = rating_color(numeric_value)
    return f"""
    <div class="rating-row">
        <div class="rating-topline">
            <span class="rating-label">{label}</span>
            <span class="rating-value">{display_value}</span>
        </div>
        <div class="rating-track">
            <div class="rating-fill" style="width: {bar_width}%; background: {bar_color};"></div>
        </div>
    </div>
    """


def render_group(title, stats, player_data):
    rows = "".join(render_stat_row(stat, player_data.get(stat)) for stat in stats)
    return f"""
    <div class="ratings-group">
        <div class="ratings-group-title">{title}</div>
        {rows}
    </div>
    """


@st.cache_data
def load_data():
    df = pd.read_csv("data/ea_cfb26_players_with_headshots.csv")
    if "Headshot URL" in df.columns:
        df["Headshot URL"] = (
            df["Headshot URL"]
            .astype(str)
            .str.extract(r'src="([^"]+)"', expand=False)
            .fillna(df["Headshot URL"])
        )
    return df


meta_df = load_data()

st.markdown(
    """
    <style>
    .player-summary {
        padding: 1rem 0 0.25rem;
        margin-top: 1rem;
    }
    .player-name {
        color: #111827;
        font-size: 2rem;
        font-weight: 800;
        line-height: 1.05;
        margin: 0.8rem 0 0;
    }
    .player-meta {
        color: #4b5563;
        font-size: 1.05rem;
        margin-top: 0.4rem;
    }
    .player-assets img {
        height: 92px;
        width: auto;
        object-fit: contain;
        filter: drop-shadow(0 6px 12px rgba(0, 0, 0, 0.25));
    }
    .asset-fallback {
        min-height: 92px;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 0 12px;
        border: 1px solid #d1d5db;
        border-radius: 12px;
        color: #6b7280;
        background: #f9fafb;
        font-size: 0.9rem;
    }
    .overview-card {
        padding: 0.5rem 0 0;
    }
    .overview-label {
        color: #6b7280;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .overview-value {
        color: #111827;
        font-size: 2.4rem;
        font-weight: 800;
        line-height: 1.1;
        margin-top: 0.2rem;
        margin-bottom: 0.75rem;
    }
    .overview-subvalue {
        color: #374151;
        font-size: 1rem;
        margin-top: 0;
        margin-bottom: 0.55rem;
    }
    .ratings-shell {
        padding: 0.5rem 0 0;
        margin-top: 0.75rem;
    }
    .ratings-group {
        margin-bottom: 1.7rem;
    }
    .ratings-group-title {
        color: #111827;
        font-size: 1.35rem;
        font-weight: 800;
        margin-bottom: 0.8rem;
    }
    .rating-row {
        margin-bottom: 1rem;
    }
    .rating-topline {
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        gap: 0.75rem;
        margin-bottom: 0.35rem;
    }
    .rating-label {
        color: #4b5563;
        font-size: 0.98rem;
        line-height: 1.2;
    }
    .rating-value {
        color: #111827;
        font-size: 1rem;
        font-weight: 700;
        min-width: 2ch;
        text-align: right;
    }
    .rating-track {
        width: 100%;
        height: 3px;
        background: #4b4b4b;
        border-radius: 999px;
        overflow: hidden;
    }
    .rating-fill {
        height: 100%;
        border-radius: 999px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.header("Player Overview")
st.write("Search a Player From the Box Below to See Their Ratings and Combine Stats")

player_options = (
    meta_df.sort_values(["Overall Rating", "Name"], ascending=[False, True])["Name"]
    .tolist()
)
player = st.selectbox("Select a player", options=player_options)
selected = meta_df.loc[meta_df["Name"] == player].iloc[0]

headshot_url = valid_image_url(selected.get("Headshot URL"))
team_logo_url = valid_image_url(selected.get("Team Logo URL"))
helmet_url = valid_image_url(selected.get("Helmet URL"))

summary_col, badge_col, info_col = st.columns([1.15, 0.45, 1.7], vertical_alignment="center")

with summary_col:
    st.markdown('<div class="player-summary">', unsafe_allow_html=True)
    if headshot_url:
        st.image(headshot_url, width=220)
    else:
        st.markdown('<div class="asset-fallback">Missing headshot</div>', unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="player-name">{selected['Name']}</div>
        <div class="player-meta">{selected['Position']} &bull; {selected['Team']}</div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with badge_col:
    st.markdown('<div class="player-summary">', unsafe_allow_html=True)
    if team_logo_url:
        st.markdown(
            f'<div class="player-assets"><img src="{team_logo_url}" alt="Team logo"></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="asset-fallback">Missing logo</div>', unsafe_allow_html=True)
    if helmet_url:
        st.markdown(
            f'<div class="player-assets"><img src="{helmet_url}" alt="Helmet"></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="asset-fallback">Missing helmet</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with info_col:
    info_left_col, info_right_col = st.columns(2)
    st.markdown(
        f"""
        <div class="overview-card">
            <div class="overview-label">Overall Rating</div>
            <div class="overview-value">{format_value(selected.get('Overall Rating')) or '--'}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with info_left_col:
        st.markdown(
            f"""
            <div class="overview-subvalue">Height: {selected.get('Height', '--')}</div>
            <div class="overview-subvalue">Jersey: {selected.get('Jersey #', '--')}</div>
            <div class="overview-subvalue">Redshirt: {selected.get('Redshirt Status', '--')}</div>
            <div class="overview-subvalue">Home State: {selected.get('Home State', '--')}</div>
            """,
            unsafe_allow_html=True,
        )
    with info_right_col:
        st.markdown(
            f"""
            <div class="overview-subvalue">Class: {selected.get('Class Year', '--')}</div>
            <div class="overview-subvalue">Conference: {selected.get('Conference', '--')}</div>
            <div class="overview-subvalue">Hometown: {selected.get('Hometown', '--')}</div>
            <div class="overview-subvalue">Team: {selected.get('Team', '--')}</div>
            """,
            unsafe_allow_html=True,
        )

top_row = ["General", "Ballcarrier", "Blocking", "Passing"]
bottom_row = ["Defense", "Receiving", "Kicking"]

st.markdown('<div class="ratings-shell">', unsafe_allow_html=True)

top_cols = st.columns(len(top_row))
for col, group_name in zip(top_cols, top_row):
    with col:
        st.markdown(
            render_group(group_name, STAT_GROUPS[group_name], selected),
            unsafe_allow_html=True,
        )

bottom_cols = st.columns(4)
for index, group_name in enumerate(bottom_row):
    with bottom_cols[index]:
        st.markdown(
            render_group(group_name, STAT_GROUPS[group_name], selected),
            unsafe_allow_html=True,
        )

with bottom_cols[3]:
    st.markdown("&nbsp;", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
