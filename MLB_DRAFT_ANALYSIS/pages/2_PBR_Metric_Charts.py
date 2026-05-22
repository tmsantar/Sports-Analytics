import altair as alt
import pandas as pd
import streamlit as st

from mlb_app_utils import load_pbr_data


st.set_page_config(page_title="PBR Metric Charts", layout="wide")

pbr_df = load_pbr_data()

st.markdown("## PBR Metric Charts")
st.caption("Weight-adjusted speed, exit velocity, hand speed, bat speed, and overall metric views.")


def chart_source(df, y_col, top_n=12):
    source = df.dropna(subset=["weight_lbs", y_col]).copy()
    source["is_dominic"] = source["player_name"].str.contains("Dominic Santarelli", case=False, na=False)
    source["label"] = ""
    top_names = source.nlargest(top_n, y_col)["player_name"]
    source.loc[source["player_name"].isin(top_names), "label"] = source["player_name"]
    source.loc[source["is_dominic"], "label"] = source["player_name"]
    return source


def quadrant_chart(df, y_col, y_title, top_n=12):
    source = chart_source(df, y_col, top_n)
    x_mid = float(source["weight_lbs"].median())
    y_mid = float(source[y_col].median())

    base = alt.Chart(source)
    points = (
        base
        .mark_circle(size=70, opacity=0.78)
        .encode(
            x=alt.X(
                "weight_lbs:Q",
                title="Weight",
                scale=alt.Scale(zero=False, nice=True, padding=20),
            ),
            y=alt.Y(
                f"{y_col}:Q",
                title=y_title,
                scale=alt.Scale(zero=False, nice=True, padding=20),
            ),
            color=alt.condition(
                alt.datum.is_dominic,
                alt.value("#d62728"),
                alt.value("#2563eb"),
            ),
            tooltip=[
                "player_name",
                "article_position",
                "height",
                "weight",
                alt.Tooltip(f"{y_col}:Q", format=".2f"),
            ],
        )
    )

    labels = (
        alt.Chart(source[source["label"] != ""])
        .mark_text(align="left", dx=7, dy=-7, fontSize=11)
        .encode(x="weight_lbs:Q", y=f"{y_col}:Q", text="label:N")
    )

    vertical = alt.Chart(pd.DataFrame({"x": [x_mid]})).mark_rule(color="#555", strokeDash=[5, 5]).encode(x="x:Q")
    horizontal = alt.Chart(pd.DataFrame({"y": [y_mid]})).mark_rule(color="#555", strokeDash=[5, 5]).encode(y="y:Q")
    trend = base.transform_regression("weight_lbs", y_col).mark_line(color="#111", strokeWidth=2).encode(
        x="weight_lbs:Q",
        y=f"{y_col}:Q",
    )

    return (points + trend + vertical + horizontal + labels).properties(width="container", height=560)


def overall_chart(df):
    source = df.dropna(subset=["rank", "overall_score"]).copy()
    source["rank"] = pd.to_numeric(source["rank"], errors="coerce")
    source = source.dropna(subset=["rank", "overall_score"])
    source["is_dominic"] = source["player_name"].str.contains("Dominic Santarelli", case=False, na=False)
    top_names = source.nlargest(12, "overall_score")["player_name"]
    source["label"] = ""
    source.loc[source["player_name"].isin(top_names), "label"] = source["player_name"]
    source.loc[source["is_dominic"], "label"] = source["player_name"]

    base = alt.Chart(source)
    points = (
        base
        .mark_circle(size=70, opacity=0.78)
        .encode(
            x=alt.X(
                "rank:Q",
                title="PBR Rank",
                sort="ascending",
                scale=alt.Scale(zero=False, nice=True, padding=20),
            ),
            y=alt.Y(
                "overall_score:Q",
                title="Overall Score",
                scale=alt.Scale(zero=False, nice=True, padding=20),
            ),
            color=alt.condition(alt.datum.is_dominic, alt.value("#d62728"), alt.value("#2563eb")),
            tooltip=[
                "player_name",
                "article_position",
                "height",
                "weight",
                alt.Tooltip("overall_score:Q", format=".2f"),
            ],
        )
    )
    labels = (
        alt.Chart(source[source["label"] != ""])
        .mark_text(align="left", dx=7, dy=-7, fontSize=11)
        .encode(x="rank:Q", y="overall_score:Q", text="label:N")
    )
    trend = base.transform_regression("rank", "overall_score").mark_line(color="#111", strokeWidth=2).encode(
        x="rank:Q",
        y="overall_score:Q",
    )
    x_mid = float(source["rank"].median())
    y_mid = float(source["overall_score"].median())
    vertical = alt.Chart(pd.DataFrame({"x": [x_mid]})).mark_rule(color="#555", strokeDash=[5, 5]).encode(x="x:Q")
    horizontal = alt.Chart(pd.DataFrame({"y": [y_mid]})).mark_rule(color="#555", strokeDash=[5, 5]).encode(y="y:Q")
    return (points + trend + vertical + horizontal + labels).properties(width="container", height=560)


with st.sidebar:
    st.markdown("### Filters")
    min_weight, max_weight = st.slider(
        "Weight range",
        int(pbr_df["weight_lbs"].min()),
        int(pbr_df["weight_lbs"].max()),
        (int(pbr_df["weight_lbs"].min()), int(pbr_df["weight_lbs"].max())),
    )
    top_labels = st.slider("Top labels", 5, 25, 12)

filtered = pbr_df[pbr_df["weight_lbs"].between(min_weight, max_weight)].copy()

tabs = st.tabs(["Speed Score", "Exit Velocity", "Hand Speed", "Bat Speed", "Overall Score"])

with tabs[0]:
    st.altair_chart(quadrant_chart(filtered, "speed_score", "Speed Score", top_labels), use_container_width=True)
    st.dataframe(
        filtered.dropna(subset=["speed_score"])
        .sort_values("speed_score", ascending=False)
        [["rank", "player_name", "article_position", "height", "weight", "sixty_time", "speed_score"]]
        .head(25),
        use_container_width=True,
        hide_index=True,
    )

with tabs[1]:
    st.altair_chart(quadrant_chart(filtered, "max_exit_velocity", "Max Exit Velocity", top_labels), use_container_width=True)
    st.dataframe(
        filtered.dropna(subset=["max_exit_velocity"])
        .sort_values("max_exit_velocity", ascending=False)
        [["rank", "player_name", "article_position", "height", "weight", "max_exit_velocity"]]
        .head(25),
        use_container_width=True,
        hide_index=True,
    )

with tabs[2]:
    st.altair_chart(quadrant_chart(filtered, "max_hand_speed", "Max Hand Speed", top_labels), use_container_width=True)
    st.dataframe(
        filtered.dropna(subset=["max_hand_speed"])
        .sort_values("max_hand_speed", ascending=False)
        [["rank", "player_name", "article_position", "height", "weight", "max_hand_speed"]]
        .head(25),
        use_container_width=True,
        hide_index=True,
    )

with tabs[3]:
    st.altair_chart(quadrant_chart(filtered, "max_bat_speed", "Max Bat Speed", top_labels), use_container_width=True)
    st.dataframe(
        filtered.dropna(subset=["max_bat_speed"])
        .sort_values("max_bat_speed", ascending=False)
        [["rank", "player_name", "article_position", "height", "weight", "max_bat_speed"]]
        .head(25),
        use_container_width=True,
        hide_index=True,
    )

with tabs[4]:
    st.altair_chart(overall_chart(filtered), use_container_width=True)
    st.dataframe(
        filtered.dropna(subset=["overall_score"])
        .sort_values("overall_score", ascending=False)
        [["rank", "player_name", "article_position", "height", "weight", "speed_score", "max_exit_velocity", "max_hand_speed", "max_bat_speed", "overall_score"]]
        .head(25),
        use_container_width=True,
        hide_index=True,
    )
