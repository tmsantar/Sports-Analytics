from pathlib import Path
import re
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
MLB_FILE = BASE_DIR / "data" / "mlb_draft" / "mlb_draft_rankings_master.csv"
PBR_FILE = BASE_DIR / "data" / "pbr" / "pbr_2026_draft_board_metrics.csv"

GRADE_COLUMNS = ["hit", "power", "run", "arm", "field", "overall"]
PBR_METRICS = ["sixty_time", "max_exit_velocity", "max_hand_speed", "max_bat_speed"]


@st.cache_data
def load_mlb_data():
    df = pd.read_csv(MLB_FILE)
    df["height_inches"] = df["height"].apply(height_inches)
    df["weight_lbs"] = pd.to_numeric(df["weight"], errors="coerce")
    for col in GRADE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # Normalize positions to eliminate duplicates like SS/OF and OF/SS
    df["position"] = df["position"].apply(normalize_position)
    # Create a primary position column (first position only)
    df["primary_pos"] = df["position"].apply(primary_position)
    return df


@st.cache_data
def load_pbr_data():
    df = pd.read_csv(PBR_FILE)
    df["weight_lbs"] = pd.to_numeric(df["weight"], errors="coerce")
    df["height_inches"] = df["height"].apply(height_inches)
    for col in PBR_METRICS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["estimated_40"] = df["sixty_time"] * (40 / 60)
    df["speed_score"] = (df["weight_lbs"] * 200) / (df["estimated_40"] ** 4)
    add_overall_score(df)
    return df


def height_inches(value):
    match = re.search(r"(\d+)\s*['-]\s*(\d+)", str(value))
    return int(match.group(1)) * 12 + int(match.group(2)) if match else np.nan


def normalize_position(position):
    """Normalize position by sorting components to eliminate duplicates (e.g., SS/OF → OF/SS)."""
    if pd.isna(position):
        return position
    # Split on /, sort components, rejoin
    components = str(position).split("/")
    normalized = "/".join(sorted(components))
    return normalized


def primary_position(position):
    """Extract the first/primary position from a position string."""
    if pd.isna(position):
        return position
    # Take the first component before any /
    return str(position).split("/")[0]


def format_number(value, digits=1):
    if pd.isna(value):
        return "N/A"
    return f"{float(value):.{digits}f}"


def format_match_percent(distance):
    match_percent = max(0, min(100, (1 - float(distance)) * 100))
    return f"{match_percent:.2f}%"


def player_options(df):
    names = df["player_name"].dropna().astype(str).sort_values().unique()
    return list(names)


def clean_positions(value):
    return set(re.split(r"[/,\s-]+", str(value).upper())) - {""}


def add_overall_score(df):
    available = df.dropna(subset=["speed_score", "max_exit_velocity", "max_hand_speed", "max_bat_speed"]).copy()
    if available.empty:
        df["overall_score"] = np.nan
        return

    df["overall_score"] = (
        df["speed_score"].rank(pct=True)
        + df["max_exit_velocity"].rank(pct=True)
        + df["max_hand_speed"].rank(pct=True)
        + df["max_bat_speed"].rank(pct=True)
    ) / 4 * 100


def build_similarity_table(df, selected_player, n_neighbors=4, hitters_only=True):
    work = df.copy()
    if hitters_only:
        work = work[~work["position"].fillna("").str.upper().str.contains("P")].copy()

    needed = ["height_inches", "weight_lbs", "primary_pos", "bats", "throws", *GRADE_COLUMNS]
    work = work.dropna(subset=needed).copy()

    selected_rows = work[work["player_name"].eq(selected_player)]
    if selected_rows.empty:
        return None, pd.DataFrame()

    feature_df = work[["height_inches", "weight_lbs", *GRADE_COLUMNS, "primary_pos", "bats", "throws"]].copy()
    feature_df["primary_pos"] = feature_df["primary_pos"].astype(str)
    feature_df["bats"] = feature_df["bats"].astype(str)
    feature_df["throws"] = feature_df["throws"].astype(str)

    encoded = pd.get_dummies(feature_df, columns=["primary_pos", "bats", "throws"])
    scaled = StandardScaler().fit_transform(encoded)

    model = NearestNeighbors(metric="cosine", algorithm="brute")
    model.fit(scaled)

    selected_index = selected_rows.index[0]
    selected_location = work.index.get_loc(selected_index)
    count = min(n_neighbors, len(work))
    distances, indices = model.kneighbors([scaled[selected_location]], n_neighbors=count)

    selected = work.loc[selected_index]
    rows = []
    for neighbor_location, distance in zip(indices.flatten(), distances.flatten()):
        neighbor = work.iloc[neighbor_location].copy()
        if neighbor["player_name"] == selected_player:
            continue
        neighbor["distance"] = float(distance)
        neighbor["match_percent"] = max(0, min(100, (1 - float(distance)) * 100))
        rows.append(neighbor)

    return selected, pd.DataFrame(rows)


def find_pbr_player(pbr_df, player_name):
    matches = pbr_df[pbr_df["player_name"].astype(str).str.lower().eq(str(player_name).lower())]
    if matches.empty:
        return None
    return matches.iloc[0]


def render_metric(label, value, digits=1):
    st.markdown(f"##### {label}: {format_number(value, digits)}")


def format_current_level(player_data):
    level = player_data.get("current_level")
    if pd.notna(level) and str(level).strip() != "":
        return str(level).strip()
    return None


def format_current_team(player_data):
    team = player_data.get("current_team")
    org = player_data.get("current_org")
    candidate = team if pd.notna(team) and str(team).strip() != "" else org
    if pd.isna(candidate) or str(candidate).strip() == "":
        return None
    return str(candidate).strip()


def format_undrafted_status(player_data):
    year = player_data.get("year")
    if pd.isna(year):
        return "No longer playing"
    current_year = datetime.now().year
    try:
        year_int = int(year)
    except (TypeError, ValueError):
        return "No longer playing"
    if year_int >= current_year:
        return "Not drafted yet"
    return "No longer playing"


def render_player_card(player_data, title, pbr_data=None, distance=None):
    st.markdown(f"### {title}")
    info_col, stats_col = st.columns([1, 2])

    with info_col:
        st.markdown(f"#### {player_data['player_name']} | {player_data['position']}")
        st.markdown(f"##### Rank: {player_data['rank']} | Year: {player_data['year']}")
        st.markdown(f"##### School: {player_data.get('school', 'N/A')}")
        current_level = format_current_level(player_data)
        current_team = format_current_team(player_data)
        if current_level:
            st.markdown(f"##### Current Level: {current_level}")
        if current_team:
            st.markdown(f"##### Current Team: {current_team}")
        if not current_level and not current_team:
            st.markdown(f"##### {format_undrafted_status(player_data)}")
        st.markdown(f"##### Size: {player_data['height']} | {player_data['weight']} lbs")
        st.markdown(f"##### B/T: {player_data.get('bats', 'N/A')}/{player_data.get('throws', 'N/A')}")
        image_url = player_data.get("profile_image_url")
        if pd.notna(image_url) and image_url:
            st.image(image_url, width=230)
        else:
            st.info("No player headshot was found")
        if pd.notna(player_data.get("profile_url")) and player_data.get("profile_url"):
            st.link_button("Open MLB Profile", player_data["profile_url"])

    with stats_col:
        col1, col2 = st.columns(2)
        with col1:
            render_metric("Hit", player_data.get("hit"), 0)
            render_metric("Power", player_data.get("power"), 0)
            render_metric("Run", player_data.get("run"), 0)
            render_metric("Arm", player_data.get("arm"), 0)
            render_metric("Field", player_data.get("field"), 0)
            render_metric("Overall", player_data.get("overall"), 0)

        with col2:
            if distance is not None:
                st.markdown(f"##### Match: {format_match_percent(distance)}")
            if pbr_data is not None:
                render_metric("60 Time", pbr_data.get("sixty_time"), 2)
                render_metric("Speed Score", pbr_data.get("speed_score"), 1)
                render_metric("Max EV", pbr_data.get("max_exit_velocity"), 1)
                render_metric("Hand Speed", pbr_data.get("max_hand_speed"), 1)
                render_metric("Bat Speed", pbr_data.get("max_bat_speed"), 1)
            else:
                st.info("No matching PBR metrics were found")

    bio = player_data.get("bio_text")
    if pd.notna(bio) and bio:
        with st.expander("Scouting bio"):
            st.write(str(bio))


def prep_comp_table(comps):
    if comps.empty:
        return comps
    table = comps.copy()
    table.insert(0, "Comp Rank", range(1, len(table) + 1))
    table["Similarity"] = table["match_percent"].map(lambda x: f"{x:.2f}%")
    table["Distance"] = table["distance"].map(lambda x: f"{x:.3f}")
    return table[
        [
            "Comp Rank",
            "player_name",
            "year",
            "rank",
            "position",
            "school",
            "current_team",
            "current_org",
            "current_level",
            "height",
            "weight",
            "bats",
            "throws",
            *GRADE_COLUMNS,
            "Similarity",
            "Distance",
        ]
    ].rename(
        columns={
            "player_name": "Player",
            "year": "Year",
            "rank": "MLB Rank",
            "position": "Pos",
            "school": "School",
            "current_team": "Cur Team",
            "current_org": "Cur Org",
            "current_level": "Level",
            "height": "Ht",
            "weight": "Wt",
            "bats": "B",
            "throws": "T",
            "hit": "Hit",
            "power": "Power",
            "run": "Run",
            "arm": "Arm",
            "field": "Field",
            "overall": "Overall",
        }
    )
