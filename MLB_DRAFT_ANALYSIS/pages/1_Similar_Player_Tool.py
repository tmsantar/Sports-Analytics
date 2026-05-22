import streamlit as st

from mlb_app_utils import (
    build_similarity_table,
    find_pbr_player,
    load_mlb_data,
    load_pbr_data,
    player_options,
    prep_comp_table,
    render_player_card,
)


st.set_page_config(page_title="Similar Player Tool", layout="wide")

mlb_df = load_mlb_data()
pbr_df = load_pbr_data()

st.markdown("## Similar Player Tool")
st.caption("Pick a draft prospect and explore the closest statistical matches.")

with st.sidebar:
    st.markdown("### Settings")
    hitters_only = st.toggle("Hitters only", value=True)
    neighbors = st.slider("Number of matches", min_value=3, max_value=10, value=3)

eligible = mlb_df.copy()
if hitters_only:
    eligible = eligible[~eligible["position"].fillna("").str.upper().str.contains("P")].copy()

names = player_options(eligible)
default_index = names.index("Dominic Santarelli") if "Dominic Santarelli" in names else 0
player = st.selectbox("Choose a player", options=names, index=default_index)

# Initialize session state for custom metrics
if "custom_metrics" not in st.session_state:
    st.session_state.custom_metrics = {}

# Use custom metrics if available, otherwise load from mlb_df
working_mlb_df = mlb_df.copy(deep=True)
if player in st.session_state.custom_metrics:
    # Explicitly update each metric for the selected player
    player_mask = working_mlb_df["player_name"] == player
    custom_data = st.session_state.custom_metrics[player]
    
    # Apply each custom metric
    for metric, value in custom_data.items():
        if metric == "position":
            working_mlb_df.loc[player_mask, "primary_pos"] = str(value)
        else:
            working_mlb_df.loc[player_mask, metric] = float(value)

selected_player, recommendations = build_similarity_table(
    working_mlb_df,
    selected_player=player,
    n_neighbors=neighbors + 1,
    hitters_only=hitters_only,
)

if selected_player is None or recommendations.empty:
    st.error("Not enough complete data for this player. Try turning off the hitter filter or choosing another player.")
    st.stop()

# Debug: Show if custom metrics are being applied
if player in st.session_state.custom_metrics:
    with st.sidebar:
        st.divider()
        st.caption("🔧 Custom Metrics Applied")
        for metric, value in st.session_state.custom_metrics[player].items():
            original = mlb_df[mlb_df["player_name"] == player][metric].values
            if len(original) > 0:
                if metric == "position":
                    st.write(f"{metric.title()}: {original[0]} → {value}")
                else:
                    st.write(f"{metric.title()}: {original[0]:.0f} → {value:.0f}")

selected_pbr = find_pbr_player(pbr_df, selected_player["player_name"])

# Custom Metrics Section
with st.expander("📊 Custom Metrics (Optional)", expanded=False):
    st.caption("Override player metrics to explore different scenarios")
    
    col1, col2, col3 = st.columns(3)
    
    # Get current custom metrics or original values
    current_custom = st.session_state.custom_metrics.get(player, {})
    original_hit = selected_player.get("hit", 50)
    original_power = selected_player.get("power", 50)
    original_run = selected_player.get("run", 50)
    original_arm = selected_player.get("arm", 50)
    original_field = selected_player.get("field", 50)
    original_overall = selected_player.get("overall", 50)
    original_position = selected_player.get("primary_pos", "")
    
    current_hit = current_custom.get("hit", original_hit)
    current_power = current_custom.get("power", original_power)
    current_run = current_custom.get("run", original_run)
    current_arm = current_custom.get("arm", original_arm)
    current_field = current_custom.get("field", original_field)
    current_overall = current_custom.get("overall", original_overall)
    current_position = current_custom.get("position", original_position)
    
    with col1:
        hit_grade = st.number_input(
            "Hit Grade",
            min_value=20.0,
            max_value=80.0,
            value=float(current_hit),
            step=1.0,
            key="custom_hit"
        )
        power_grade = st.number_input(
            "Power Grade",
            min_value=20.0,
            max_value=80.0,
            value=float(current_power),
            step=1.0,
            key="custom_power"
        )
    
    with col2:
        run_grade = st.number_input(
            "Run Grade",
            min_value=20.0,
            max_value=80.0,
            value=float(current_run),
            step=1.0,
            key="custom_run"
        )
        arm_grade = st.number_input(
            "Arm Grade",
            min_value=20.0,
            max_value=80.0,
            value=float(current_arm),
            step=1.0,
            key="custom_arm"
        )
    
    with col3:
        field_grade = st.number_input(
            "Field Grade",
            min_value=20.0,
            max_value=80.0,
            value=float(current_field),
            step=1.0,
            key="custom_field"
        )
        overall_grade = st.number_input(
            "Overall Grade",
            min_value=20.0,
            max_value=80.0,
            value=float(current_overall),
            step=1.0,
            key="custom_overall"
        )
    
    # Position selector
    unique_positions = sorted(mlb_df["primary_pos"].unique())
    position_value = st.selectbox(
        "Position",
        options=unique_positions,
        index=unique_positions.index(str(current_position)) if str(current_position) in unique_positions else 0,
        key="custom_position"
    )
    
    # Show original vs current values
    if current_custom:
        st.caption("**Original → Current**")
        metric_cols = st.columns(4)
        metric_cols[0].write(f"**Hit**\n{original_hit:.0f} → {hit_grade:.0f}")
        metric_cols[1].write(f"**Power**\n{original_power:.0f} → {power_grade:.0f}")
        metric_cols[2].write(f"**Overall**\n{original_overall:.0f} → {overall_grade:.0f}")
        metric_cols[3].write(f"**Position**\n{original_position} → {position_value}")
    
    # Check if any custom metrics differ from original
    custom_metrics_changed = (
        hit_grade != original_hit or
        power_grade != original_power or
        run_grade != original_run or
        arm_grade != original_arm or
        field_grade != original_field or
        overall_grade != original_overall or
        position_value != original_position
    )
    
    if custom_metrics_changed:
        st.warning("⚠️ Using custom metrics — click below to recalculate comparisons")
        if st.button("🔄 Recalculate with Custom Metrics", key="recalc_button"):
            # Store custom metrics in session state
            st.session_state.custom_metrics[player] = {
                "hit": hit_grade,
                "power": power_grade,
                "run": run_grade,
                "arm": arm_grade,
                "field": field_grade,
                "overall": overall_grade,
                "position": position_value,
            }
            st.rerun()
    else:
        if st.button("🔄 Clear Custom Metrics", key="clear_button"):
            if player in st.session_state.custom_metrics:
                del st.session_state.custom_metrics[player]
            st.rerun()

st.markdown("---")
render_player_card(selected_player, "Selected Player", selected_pbr)
st.markdown("---")

st.subheader(f"Top {min(neighbors, len(recommendations))} Similar Players to {player}")
table = prep_comp_table(recommendations.head(neighbors))
st.dataframe(table, use_container_width=True, hide_index=True)

tabs = st.tabs(
    [f"#{i} {rec['player_name']}" for i, (_, rec) in enumerate(recommendations.head(neighbors).iterrows(), start=1)]
)

for i, (tab, (_, rec)) in enumerate(zip(tabs, recommendations.head(neighbors).iterrows()), start=1):
    with tab:
        pbr_match = find_pbr_player(pbr_df, rec["player_name"])
        render_player_card(
            rec,
            f"Match #{i} | {max(0, min(100, (1 - float(rec['distance'])) * 100)):.2f}% Match",
            pbr_match,
            rec["distance"],
        )

csv = table.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download visible comps",
    data=csv,
    file_name=f"{player.lower().replace(' ', '_')}_knn_comps.csv",
    mime="text/csv",
)
