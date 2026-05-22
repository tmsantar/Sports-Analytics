from pathlib import Path
import re

import numpy as np
import pandas as pd


PLAYER = "Dominic Santarelli"
BASE = Path(__file__).resolve().parent
MLB = BASE / "data/mlb_draft/mlb_draft_rankings_master.csv"
OUT = BASE / "data/mlb_draft/dominic_santarelli_knn_comps.csv"
OUT_HTML = BASE / "data/mlb_draft/dominic_santarelli_knn_comps_table.html"


def height_inches(value):
    match = re.search(r"(\d+)\s*['-]\s*(\d+)", str(value))
    return int(match.group(1)) * 12 + int(match.group(2)) if match else np.nan


def positions(value):
    return set(re.split(r"[/,\s-]+", str(value).upper())) - {""}


def clean_number(value, digits=1):
    if pd.isna(value):
        return ""
    return round(float(value), digits)


def print_table(table):
    widths = {col: max(len(col), table[col].astype(str).str.len().max()) for col in table.columns}
    header = " | ".join(col.ljust(widths[col]) for col in table.columns)
    line = "-+-".join("-" * widths[col] for col in table.columns)
    print(header)
    print(line)
    for _, row in table.iterrows():
        print(" | ".join(str(row[col]).ljust(widths[col]) for col in table.columns))


df = pd.read_csv(MLB)
df["height_inches"] = df["height"].apply(height_inches)
df["weight_lbs"] = pd.to_numeric(df["weight"], errors="coerce")

grade_cols = ["hit", "power", "run", "arm", "field", "overall"]
for col in grade_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

needed = ["height_inches", "weight_lbs", "position", "bats", "throws", *grade_cols]
df = df.dropna(subset=needed).copy()

target = df[df["player_name"].str.contains(PLAYER, case=False, na=False)].iloc[0]
target_pos = positions(target["position"])

for col in ["height_inches", "weight_lbs", *grade_cols]:
    std = df[col].std()
    df[f"{col}_diff"] = (df[col] - target[col]) / std if std else 0

df["position_diff"] = df["position"].apply(lambda x: 0 if positions(x) & target_pos else 1)
df["bats_diff"] = (df["bats"] != target["bats"]).astype(int)
df["throws_diff"] = (df["throws"] != target["throws"]).astype(int)

df["knn_distance"] = np.sqrt(
    (1.25 * df["height_inches_diff"]) ** 2
    + (1.25 * df["weight_lbs_diff"]) ** 2
    + (1.75 * df["position_diff"]) ** 2
    + (0.50 * df["bats_diff"]) ** 2
    + (0.50 * df["throws_diff"]) ** 2
    + sum((df[f"{col}_diff"]) ** 2 for col in grade_cols)
)
df["similarity_score"] = 100 / (1 + df["knn_distance"])

same_player = df["player_name"].str.contains(PLAYER, case=False, na=False)
cols = [
    "similarity_score",
    "knn_distance",
    "year",
    "rank",
    "player_name",
    "position",
    "school",
    "height",
    "weight",
    "bats",
    "throws",
    "hit",
    "power",
    "run",
    "arm",
    "field",
    "overall",
    "profile_url",
]

comps = df.loc[~same_player].sort_values("knn_distance")[cols].head(25)
OUT.parent.mkdir(parents=True, exist_ok=True)
comps.to_csv(OUT, index=False)

display = comps.reset_index(drop=True).copy()
display.insert(0, "comp_rank", display.index + 1)
display["similarity_score"] = display["similarity_score"].apply(lambda x: clean_number(x, 1))
display["knn_distance"] = display["knn_distance"].apply(lambda x: clean_number(x, 2))
display = display[
    [
        "comp_rank",
        "player_name",
        "year",
        "rank",
        "position",
        "height",
        "weight",
        "bats",
        "throws",
        "hit",
        "power",
        "run",
        "arm",
        "field",
        "overall",
        "similarity_score",
        "knn_distance",
    ]
].rename(
    columns={
        "comp_rank": "#",
        "player_name": "Player",
        "year": "Year",
        "rank": "MLB Rank",
        "position": "Pos",
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
        "similarity_score": "Similarity",
        "knn_distance": "Distance",
    }
)

html = display.to_html(index=False, border=0, classes="comps-table")
OUT_HTML.write_text(
    f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Dominic Santarelli KNN Comps</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 28px; color: #111; }}
h1 {{ font-size: 24px; margin: 0 0 6px; }}
p {{ margin: 0 0 18px; color: #444; }}
table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
th, td {{ padding: 8px 10px; border-bottom: 1px solid #ddd; text-align: left; }}
th {{ background: #111; color: white; position: sticky; top: 0; }}
tr:nth-child(even) {{ background: #f7f7f7; }}
td:first-child, th:first-child {{ text-align: right; }}
</style>
</head>
<body>
<h1>Dominic Santarelli KNN Comps</h1>
<p>Target: rank {target['rank']} | {target['position']} | {target['height']} | {target['weight']} lbs | grades {target['hit']}/{target['power']}/{target['run']}/{target['arm']}/{target['field']}/{target['overall']}</p>
{html}
</body>
</html>
""",
    encoding="utf-8",
)

print(f"Target from MLB file: {target['player_name']} | rank {target['rank']} | {target['position']} | {target['height']} | {target['weight']}")
print(f"Grades: hit {target['hit']}, power {target['power']}, run {target['run']}, arm {target['arm']}, field {target['field']}, overall {target['overall']}")
print(f"Saved CSV: {OUT}")
print(f"Saved clean table: {OUT_HTML}")
print()
print_table(display)
