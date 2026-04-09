"""
Scrape MiLB draft prospect rankings by year and extract scouting grades.

This targets pages like:
https://www.mlb.com/milb/prospects/2025/draft/
"""

from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
import requests


BASE_URL = "https://www.mlb.com/milb/prospects/{year}/draft/"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/135.0.0.0 Safari/537.36"
)


def fetch_html(year: int, session: requests.Session) -> str:
    url = BASE_URL.format(year=year)
    try:
        response = session.get(url, timeout=30)
        response.raise_for_status()
        return response.text
    except requests.RequestException:
        command = (
            "$ProgressPreference='SilentlyContinue'; "
            f"$r = Invoke-WebRequest -Uri '{url}' -UseBasicParsing; "
            "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; "
            "Write-Output $r.Content"
        )
        completed = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                command,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
        )
        return completed.stdout


def page_has_rankings(html_text: str, year: int) -> bool:
    return f"sel-pr-{year}-draft" in html_text


def discover_years(session: requests.Session, start_year: int, end_year: int) -> list[int]:
    years: list[int] = []
    for year in range(start_year, end_year + 1):
        try:
            html_text = fetch_html(year, session)
        except requests.RequestException:
            continue
        if page_has_rankings(html_text, year):
            years.append(year)
    return years


def find_matching_bracket(text: str, start_index: int, open_char: str, close_char: str) -> int:
    depth = 0
    in_string = False
    escape = False

    for index in range(start_index, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == open_char:
            depth += 1
        elif char == close_char:
            depth -= 1
            if depth == 0:
                return index

    raise ValueError("Could not find matching bracket.")


def extract_json_segment(text: str, key: str, opening_char: str) -> str:
    key_index = text.find(key)
    if key_index == -1:
        raise ValueError(f"Could not find key: {key}")

    start_index = text.find(opening_char, key_index)
    if start_index == -1:
        raise ValueError(f"Could not find opening char for key: {key}")

    closing_char = "]" if opening_char == "[" else "}"
    end_index = find_matching_bracket(text, start_index, opening_char, closing_char)
    return text[start_index : end_index + 1]


def unescape_page(html_text: str) -> str:
    return html.unescape(html_text)


def extract_rankings(html_text: str, year: int) -> list[dict[str, Any]]:
    unescaped = unescape_page(html_text)
    candidate_keys = [
        f'"getPlayerRankingsFromSelection({{\\"limit\\":100,\\"skip\\":0,\\"slug\\":\\"sel-pr-{year}-draft\\"}})"',
        f'"getPlayerRankingsFromSelection({{"limit":100,"skip":0,"slug":"sel-pr-{year}-draft"}})"',
        f'"getPlayerRankingsFromSelection({{\\"limit\\":100,\\"slug\\":\\"sel-pr-{year}-draft\\"}})"',
        f'"getPlayerRankingsFromSelection({{"limit":100,"slug":"sel-pr-{year}-draft"}})"',
    ]

    for key in candidate_keys:
        try:
            array_text = extract_json_segment(unescaped, key, "[")
            return json.loads(array_text)
        except (ValueError, json.JSONDecodeError):
            continue

    raise ValueError(f"Could not extract draft rankings payload for {year}.")


def extract_object_by_key(unescaped_html: str, key: str) -> dict[str, Any]:
    key_pattern = f"{key}:{{"
    key_index = unescaped_html.find(key_pattern)
    if key_index == -1:
        raise ValueError(f"Could not find object definition for key: {key}")

    start_index = unescaped_html.find("{", key_index)
    end_index = find_matching_bracket(unescaped_html, start_index, "{", "}")
    return json.loads(unescaped_html[start_index : end_index + 1])


def strip_html_tags(text: str | None) -> str | None:
    if not text:
        return None
    text = html.unescape(text)
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def normalize_grade_label(label: str) -> str:
    label = label.strip().lower().replace("*", "")
    aliases = {"fb": "fastball", "curve": "curveball"}
    label = aliases.get(label, label)
    return re.sub(r"[^a-z0-9]+", "_", label).strip("_")


def extract_grade_text(bio_html: str | None) -> str | None:
    if not bio_html:
        return None

    html_match = re.search(
        r"Scouting grades?\*?(?:\s*\(present/future\))?:\s*</strong>\s*(.*?)</p>",
        bio_html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if html_match:
        return strip_html_tags(html_match.group(1))

    bio_text = strip_html_tags(bio_html)
    if not bio_text:
        return None

    text_match = re.search(
        r"scouting grades?\*?(?:\s*\(present/future\))?:\s*(.+?)(?:\.\s+[A-Z]|$)",
        bio_text,
        flags=re.IGNORECASE,
    )
    if not text_match:
        return None

    grade_text = text_match.group(1).strip()
    if grade_text.endswith("."):
        grade_text = grade_text[:-1].strip()
    return grade_text or None


def parse_grades(grade_text: str | None) -> dict[str, str]:
    if not grade_text:
        return {}

    grades: dict[str, str] = {}
    for part in grade_text.split("|"):
        if ":" not in part:
            continue
        label, value = part.split(":", 1)
        grades[normalize_grade_label(label)] = value.strip()
    return grades


def find_relevant_bio(prospect_bios: list[dict[str, Any]], year: int) -> dict[str, Any] | None:
    preferred_titles = {str(year), "draft"}
    for bio in prospect_bios or []:
        title = str(bio.get("contentTitle", "")).strip().lower()
        if title in preferred_titles:
            return bio
    return (prospect_bios or [None])[0]


def extract_structured_grades(player_entity: dict[str, Any]) -> dict[str, str]:
    grades: dict[str, str] = {}
    for grade_group in ("gradesHitting", "gradesPitching"):
        for item in player_entity.get(grade_group) or []:
            key = item.get("key")
            value = item.get("value")
            if key and value is not None:
                grades[normalize_grade_label(str(key))] = str(value).strip()
    return grades


def extract_row_metadata(html_text: str) -> dict[int, dict[str, str | None]]:
    metadata: dict[int, dict[str, str | None]] = {}

    row_pattern = re.compile(r"<tr\b[^>]*data-testid=\"table-row\"[^>]*>(.*?)</tr>", re.IGNORECASE | re.DOTALL)
    rank_pattern = re.compile(r"rankings__table__cell--rank\"[^>]*>\s*(\d+)\s*</td>", re.IGNORECASE | re.DOTALL)
    school_pattern = re.compile(r"rankings__table__cell--schoolName\"[^>]*>\s*([^<]+?)\s*</td>", re.IGNORECASE | re.DOTALL)
    level_pattern = re.compile(r"prospect-level__label\"[^>]*>\s*([^<]+?)\s*</div>", re.IGNORECASE | re.DOTALL)
    image_pattern = re.compile(r"<img[^>]+data-testid=\"player-headshot\"[^>]+src=\"([^\"]+)\"", re.IGNORECASE | re.DOTALL)

    for row_html in row_pattern.findall(html_text):
        rank_match = rank_pattern.search(row_html)
        if not rank_match:
            continue

        rank = int(rank_match.group(1))
        school_match = school_pattern.search(row_html)
        level_match = level_pattern.search(row_html)
        image_match = image_pattern.search(row_html)
        metadata[rank] = {
            "school": school_match.group(1).strip() if school_match else None,
            "current_level": level_match.group(1).strip() if level_match else None,
            "profile_image_url": image_match.group(1).strip() if image_match else None,
        }

    return metadata


def get_school_from_person(person: dict[str, Any]) -> str | None:
    education = person.get("education") or {}
    colleges = education.get("colleges") or []
    highschools = education.get("highschools") or []

    if colleges:
        return colleges[0].get("name")
    if highschools:
        hs = highschools[0]
        name = hs.get("name")
        state = hs.get("state")
        if name and state:
            return f"{name} ({state})"
        return name
    return None


def build_year_dataframe(year: int, html_text: str) -> pd.DataFrame:
    unescaped = unescape_page(html_text)
    rankings = extract_rankings(html_text, year)
    row_metadata = extract_row_metadata(html_text)
    rows: list[dict[str, Any]] = []

    for item in rankings:
        player_entity = item.get("playerEntity") or {}
        player_ref = ((player_entity.get("player") or {}).get("__ref")) or ""
        person_id = player_ref.split(":", 1)[1] if ":" in player_ref else None
        person = extract_object_by_key(unescaped, f'"Person:{person_id}"') if person_id else {}

        active_team_ref = ((person.get("activeRoster") or {}).get("__ref")) or ((person.get("activeRosterMLB") or {}).get("__ref"))
        team = extract_object_by_key(unescaped, f'"{active_team_ref}"') if active_team_ref else {}

        drafts = person.get("drafts") or []
        first_draft = drafts[0] if drafts else {}
        draft_team_ref = ((first_draft.get("team") or {}).get("__ref")) or ""
        draft_team = extract_object_by_key(unescaped, f'"{draft_team_ref}"') if draft_team_ref else {}

        bio = find_relevant_bio(player_entity.get("prospectBio") or [], year) or {}
        bio_html = bio.get("contentText")
        bio_text = strip_html_tags(bio_html)
        grade_text = extract_grade_text(bio_html)
        grade_values = extract_structured_grades(player_entity) or parse_grades(grade_text)
        row_meta = row_metadata.get(item.get("rank")) or {}

        row = {
            "year": year,
            "rank": item.get("rank"),
            "player_id": person.get("id"),
            "player_name": " ".join(part for part in [person.get("useName"), person.get("useLastName")] if part) or None,
            "name_slug": person.get("nameSlug"),
            "profile_url": (
                f"https://www.mlb.com/milb/prospects/{year}/draft/{person.get('nameSlug')}"
                if person.get("nameSlug")
                else None
            ),
            "position": player_entity.get("position") or ((person.get("primaryPosition") or {}).get("abbreviation")),
            "school": row_meta.get("school") or get_school_from_person(person),
            "current_team": team.get("name"),
            "current_org": team.get("parentOrgName"),
            "current_level": row_meta.get("current_level"),
            "current_age": person.get("currentAge"),
            "bats": person.get("batSideCode"),
            "throws": person.get("pitchHandCode"),
            "birth_date": person.get("birthDate"),
            "birth_country": person.get("birthCountry"),
            "height": person.get("height"),
            "weight": person.get("weight"),
            "profile_image_url": player_entity.get("playerPhotoCustomUrl") or row_meta.get("profile_image_url"),
            "eta": player_entity.get("eta"),
            "player_draft_year": first_draft.get("year"),
            "player_draft_round": first_draft.get("pickRound"),
            "player_draft_pick_number": first_draft.get("pickNumber"),
            "player_draft_team_id": draft_team.get("id"),
            "player_draft_team_name": draft_team.get("name"),
            "scouting_grades_text": grade_text or " | ".join(
                f"{key.replace('_', ' ').title()}: {value}" for key, value in grade_values.items()
            ),
            "bio_text": bio_text,
        }
        if first_draft.get("year") is not None:
            try:
                row["years_since_draft"] = year - int(first_draft.get("year"))
            except (TypeError, ValueError):
                row["years_since_draft"] = None
        else:
            row["years_since_draft"] = None
        row.update(grade_values)
        rows.append(row)

    if not rows:
        raise ValueError(f"No rows parsed for draft year {year}.")

    return pd.DataFrame(rows).sort_values(["rank", "player_name"]).reset_index(drop=True)


def scrape_all_years(start_year: int = 2015, end_year: int = 2026) -> tuple[dict[int, pd.DataFrame], pd.DataFrame]:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    years = discover_years(session, start_year, end_year)
    if not years:
        raise RuntimeError("No draft ranking years were discovered.")

    year_frames: dict[int, pd.DataFrame] = {}
    for year in years:
        html_text = fetch_html(year, session)
        try:
            year_frames[year] = build_year_dataframe(year, html_text)
        except Exception as exc:
            raise RuntimeError(f"Failed while parsing draft year {year}.") from exc

    combined = pd.concat(year_frames.values(), ignore_index=True, sort=False)
    return year_frames, combined


def save_outputs(year_frames: dict[int, pd.DataFrame], combined: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(outdir / "mlb_draft_rankings_all_years.csv", index=False)
    for year, frame in year_frames.items():
        frame.to_csv(outdir / f"mlb_draft_rankings_{year}.csv", index=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scrape MiLB draft prospect rankings and scouting grades.")
    parser.add_argument("--start-year", type=int, default=2015)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--outdir", type=Path, default=Path("data/mlb_draft"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    year_frames, combined = scrape_all_years(args.start_year, args.end_year)
    save_outputs(year_frames, combined, args.outdir)

    years = sorted(year_frames)
    print(f"Saved {len(years)} yearly files plus one combined file to: {args.outdir}")
    print(f"Years scraped: {years[0]}-{years[-1]}")
    print(f"Combined rows: {len(combined):,}")


if __name__ == "__main__":
    main()
