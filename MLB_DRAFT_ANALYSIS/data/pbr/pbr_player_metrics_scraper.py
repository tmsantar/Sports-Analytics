"""
Scrape Prep Baseball Report draft-board players and profile metrics.

Default source:
https://www.prepbaseballreport.com/news/PBR/2026-mlb-draft-board--top-200--spring-update#DraftBoard

The scraper first finds player/profile links from the draft-board article, then
visits each profile and extracts whichever bio and showcase metrics are present.
PBR profiles are not perfectly uniform, so missing values are left blank.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests

try:
    from bs4 import BeautifulSoup
except ImportError:  # BeautifulSoup helps, but the regex fallbacks keep this script portable.
    BeautifulSoup = None


DEFAULT_DRAFT_BOARD_URL = (
    "https://www.prepbaseballreport.com/news/PBR/"
    "2026-mlb-draft-board--top-200--spring-update#DraftBoard"
)
BASE_URL = "https://www.prepbaseballreport.com"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/136.0.0.0 Safari/537.36"
)

METRIC_LABELS = {
    "sixty_time": [
        "60 Time",
        "60 Yard Dash",
        "60-Yard Dash",
        "VALD 60 Time",
        "Lasered 60 Time",
    ],
    "home_to_first": ["Home to First", "H-1st", "Home-1st"],
    "max_exit_velocity": [
        "Max Exit Velocity",
        "Exit Velocity",
        "Exit Velo",
        "EV Max",
        "Max EV",
    ],
    "avg_exit_velocity": ["Avg Exit Velocity", "Average Exit Velocity", "Avg EV"],
    "max_hand_speed": ["Max Hand Speed", "Hand Speed"],
    "max_bat_speed": ["Max Bat Speed", "Bat Speed"],
    "rotational_acceleration": ["Rotational Acceleration", "Rot. Acceleration"],
    "infield_velocity": ["Infield Velocity", "INF Velo", "INF Velocity"],
    "outfield_velocity": ["Outfield Velocity", "OF Velo", "OF Velocity"],
    "catcher_velocity": ["Catcher Velocity", "C Velo", "C Velocity"],
    "pop_time": ["Pop Time"],
    "fastball_velocity": ["Fastball", "FB", "FB Velo", "Max FB"],
    "curveball_velocity": ["Curveball", "CB"],
    "slider_velocity": ["Slider", "SL"],
    "changeup_velocity": ["Changeup", "CH"],
}

STATE_IDS = {
    "AL": "31",
    "AK": "32",
    "AZ": "33",
    "AR": "20",
    "CA": "34",
    "CO": "13",
    "CT": "28",
    "DE": "35",
    "FL": "16",
    "GA": "36",
    "HI": "37",
    "ID": "38",
    "IL": "2",
    "IN": "3",
    "IA": "14",
    "KS": "11",
    "KY": "8",
    "LA": "22",
    "ME": "25",
    "MD": "23",
    "MA": "21",
    "MI": "6",
    "MN": "15",
    "MS": "39",
    "MO": "4",
    "MT": "40",
    "NE": "41",
    "NV": "42",
    "NH": "27",
    "NJ": "30",
    "NM": "43",
    "NY": "17",
    "NC": "44",
    "ND": "45",
    "OH": "5",
    "OK": "46",
    "OR": "47",
    "PA": "7",
    "RI": "29",
    "SC": "48",
    "SD": "49",
    "TN": "18",
    "TX": "50",
    "UT": "51",
    "VT": "26",
    "VA": "12",
    "WA": "52",
    "WV": "10",
    "WI": "9",
    "WY": "53",
}

MANUAL_PROFILE_URLS = {
    "josephlawson": "https://www.prepbaseballreport.com/profiles/FL/Joseph-Lawson",
}


@dataclass
class PlayerLink:
    rank: int | None
    player_name: str | None
    profile_url: str | None
    article_position: str | None = None
    article_school: str | None = None
    article_state: str | None = None
    article_notes: str | None = None


def build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": BASE_URL,
        }
    )
    return session


def fetch_html(url: str, session: requests.Session) -> str:
    try:
        response = session.get(url, timeout=35)
        response.raise_for_status()
        return response.text
    except requests.RequestException:
        command = (
            "$ProgressPreference='SilentlyContinue'; "
            f"$r = Invoke-WebRequest -Uri '{url}' -UseBasicParsing "
            f"-Headers @{{'User-Agent'='{USER_AGENT}'; 'Referer'='{BASE_URL}'}}; "
            "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; "
            "Write-Output $r.Content"
        )
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
        )
        return completed.stdout


def clean_text(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = html.unescape(value)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or None


def slug_to_name(profile_url: str) -> str | None:
    path = urlparse(profile_url).path.strip("/")
    slug = path.split("/")[-1] if path else ""
    slug = re.sub(r"-\d+$", "", slug)
    slug = slug.replace("-", " ")
    return slug.title() if slug else None


def normalize_player_name(name: str | None) -> str | None:
    if not name:
        return None
    name = html.unescape(name)
    name = re.sub(r"[🔺🆕★*†‡]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name or None


def name_key(name: str | None) -> str:
    name = normalize_player_name(name) or ""
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def infer_state_from_school(school: str | None) -> str | None:
    if not school:
        return None
    match = re.search(r"\(([A-Z]{2})\)\s*$", school)
    return match.group(1) if match else None


def normalize_profile_url(href: str, source_url: str) -> str | None:
    url = urljoin(source_url, href)
    parsed = urlparse(url)
    if "prepbaseballreport.com" not in parsed.netloc:
        return None
    if "/profiles/" not in parsed.path:
        return None
    return parsed._replace(fragment="", query="").geturl()


def parse_rank(text: str) -> int | None:
    match = re.search(r"\b(\d{1,3})\b", text)
    return int(match.group(1)) if match else None


def parse_draft_board(html_text: str, source_url: str) -> list[PlayerLink]:
    if BeautifulSoup is None:
        return parse_draft_board_with_regex(html_text, source_url)

    soup = BeautifulSoup(html_text, "html.parser")
    players: list[PlayerLink] = []
    profile_links = extract_article_profile_links(html_text, source_url)
    article_notes = extract_article_notes(html_text)
    draft_anchor = soup.find(id="DraftBoard")
    table = draft_anchor.find_next("table", class_="results-table") if draft_anchor else soup.select_one("table.results-table")
    if table:
        for row in table.find_all("tr"):
            cells = [clean_text(cell.get_text(" ")) for cell in row.find_all(["td", "th"])]
            if len(cells) < 4 or not (cells[0] or "").isdigit():
                continue
            player_name = normalize_player_name(cells[1])
            state = infer_state_from_school(cells[3])
            key = name_key(player_name)
            players.append(
                PlayerLink(
                    rank=int(cells[0]),
                    player_name=player_name,
                    profile_url=profile_links.get(key),
                    article_position=cells[2],
                    article_school=cells[3],
                    article_state=state,
                    article_notes=article_notes.get(key),
                )
            )

    if players:
        return players

    raise ValueError("No PBR profile links were found in the draft-board HTML.")


def parse_draft_board_with_regex(html_text: str, source_url: str) -> list[PlayerLink]:
    profile_links = extract_article_profile_links(html_text, source_url)
    article_notes = extract_article_notes(html_text)
    table_match = re.search(
        r'<a\s+id=["\']DraftBoard["\'][^>]*>.*?<table\b[^>]*class=["\'][^"\']*results-table[^"\']*["\'][^>]*>(.*?)</table>',
        html_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    table_html = table_match.group(1) if table_match else html_text
    players: list[PlayerLink] = []
    for row_html in re.findall(r"<tr\b[^>]*>(.*?)</tr>", table_html, flags=re.IGNORECASE | re.DOTALL):
        cells = [
            clean_text(re.sub(r"<[^>]+>", " ", cell_html))
            for cell_html in re.findall(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", row_html, flags=re.IGNORECASE | re.DOTALL)
        ]
        if len(cells) < 4 or not (cells[0] or "").isdigit():
            continue
        player_name = normalize_player_name(cells[1])
        key = name_key(player_name)
        players.append(
            PlayerLink(
                rank=int(cells[0]),
                player_name=player_name,
                profile_url=profile_links.get(key),
                article_position=cells[2],
                article_school=cells[3],
                article_state=infer_state_from_school(cells[3]),
                article_notes=article_notes.get(key),
            )
        )

    if not players:
        raise ValueError("No ranked draft-board table rows were found in the article HTML.")
    return players


def extract_article_profile_links(html_text: str, source_url: str) -> dict[str, str]:
    links: dict[str, str] = {}
    anchor_pattern = re.compile(
        r"<a\b[^>]*href=[\"']([^\"']*?/profiles/[^\"']+)[\"'][^>]*>(.*?)</a>",
        flags=re.IGNORECASE | re.DOTALL,
    )
    for href, anchor_html in anchor_pattern.findall(html_text):
        profile_url = normalize_profile_url(href, source_url)
        player_name = clean_text(re.sub(r"<[^>]+>", " ", anchor_html))
        if profile_url and player_name:
            links[name_key(player_name)] = profile_url
    return links


def extract_article_notes(html_text: str) -> dict[str, str]:
    notes: dict[str, str] = {}
    paragraph_pattern = re.compile(r"<p\b[^>]*>(.*?)</p>", flags=re.IGNORECASE | re.DOTALL)
    link_pattern = re.compile(
        r"<a\b[^>]*href=[\"'][^\"']*?/profiles/[^\"']+[\"'][^>]*>(.*?)</a>",
        flags=re.IGNORECASE | re.DOTALL,
    )
    for paragraph_html in paragraph_pattern.findall(html_text):
        paragraph_text = clean_text(re.sub(r"<[^>]+>", " ", paragraph_html))
        if not paragraph_text:
            continue
        for link_html in link_pattern.findall(paragraph_html):
            player_name = clean_text(re.sub(r"<[^>]+>", " ", link_html))
            if player_name:
                notes[name_key(player_name)] = paragraph_text
    return notes


def load_player_links_from_csv(path: Path) -> list[PlayerLink]:
    players: list[PlayerLink] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            profile_url = row.get("profile_url") or row.get("url")
            if not profile_url:
                continue
            rank_value = row.get("rank")
            players.append(
                PlayerLink(
                    rank=int(rank_value) if rank_value and rank_value.isdigit() else None,
                    player_name=row.get("player_name") or row.get("name") or slug_to_name(profile_url),
                    profile_url=profile_url,
                    article_position=row.get("position"),
                    article_school=row.get("school"),
                    article_state=row.get("state"),
                )
            )
    return players


def fetch_profile_search(
    player: PlayerLink,
    session: requests.Session,
    source_url: str,
    class_year: str | None,
) -> str:
    state_id = STATE_IDS.get(player.article_state or "")
    attempts = [
        {"player_name": player.player_name or "", "player_state": state_id or "#", "player_class": class_year or "#"},
        {"player_name": player.player_name or "", "player_state": state_id or "#", "player_class": "#"},
        {"player_name": player.player_name or "", "player_state": "#", "player_class": class_year or "#"},
        {"player_name": player.player_name or "", "player_state": "#", "player_class": "#"},
    ]
    last_html = ""
    for body in attempts:
        body.update({"player_school": "", "player_position": "#", "value_search_go": "Search"})
        response = session.post(
            urljoin(source_url, "/profile-search-results"),
            data=body,
            timeout=35,
            headers={"Referer": source_url},
        )
        response.raise_for_status()
        last_html = response.text
        if "/profiles/" in last_html:
            return last_html
    return last_html


def parse_profile_search_results(html_text: str, source_url: str, player: PlayerLink) -> str | None:
    candidates: list[dict[str, str | None]] = []
    row_pattern = re.compile(r"<tr\b[^>]*>(.*?)</tr>", flags=re.IGNORECASE | re.DOTALL)
    cell_pattern = re.compile(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", flags=re.IGNORECASE | re.DOTALL)
    link_pattern = re.compile(
        r"<a\b[^>]*href=[\"']([^\"']*?/profiles/[^\"']+)[\"'][^>]*>(.*?)</a>",
        flags=re.IGNORECASE | re.DOTALL,
    )

    for row_html in row_pattern.findall(html_text):
        link_match = link_pattern.search(row_html)
        if not link_match:
            continue
        cells = [clean_text(re.sub(r"<[^>]+>", " ", cell)) for cell in cell_pattern.findall(row_html)]
        candidates.append(
            {
                "profile_url": normalize_profile_url(link_match.group(1), source_url),
                "name": clean_text(re.sub(r"<[^>]+>", " ", link_match.group(2))),
                "state": cells[1] if len(cells) > 1 else None,
                "school": cells[2] if len(cells) > 2 else None,
                "class": cells[3] if len(cells) > 3 else None,
                "position": cells[4] if len(cells) > 4 else None,
            }
        )

    if not candidates:
        return None

    target_name = name_key(player.player_name)
    target_state = player.article_state
    target_school = re.sub(r"\s*\([A-Z]{2}\)\s*$", "", player.article_school or "").lower()
    target_position = (player.article_position or "").split("/")[0].strip().upper()

    def score(candidate: dict[str, str | None]) -> int:
        candidate_score = 0
        if name_key(candidate.get("name")) == target_name:
            candidate_score += 100
        if target_state and candidate.get("state") == target_state:
            candidate_score += 20
        if target_school and target_school in (candidate.get("school") or "").lower():
            candidate_score += 10
        if target_position and target_position in (candidate.get("position") or ""):
            candidate_score += 5
        return candidate_score

    best = max(candidates, key=score)
    return best.get("profile_url") if score(best) >= 80 else None


def resolve_missing_profile_urls(
    players: list[PlayerLink],
    session: requests.Session,
    source_url: str,
    class_year: str | None,
    delay: float,
) -> list[PlayerLink]:
    resolved: list[PlayerLink] = []
    for index, player in enumerate(players, start=1):
        manual_url = MANUAL_PROFILE_URLS.get(name_key(player.player_name))
        if manual_url and not player.profile_url:
            player = PlayerLink(
                rank=player.rank,
                player_name=player.player_name,
                profile_url=manual_url,
                article_position=player.article_position,
                article_school=player.article_school,
                article_state=player.article_state,
                article_notes=player.article_notes,
            )
        if player.profile_url:
            resolved.append(player)
            continue
        print(f"[resolve {index}/{len(players)}] {player.player_name}")
        try:
            search_html = fetch_profile_search(player, session, source_url, class_year)
            profile_url = parse_profile_search_results(search_html, source_url, player)
            player = PlayerLink(
                rank=player.rank,
                player_name=player.player_name,
                profile_url=profile_url,
                article_position=player.article_position,
                article_school=player.article_school,
                article_state=player.article_state,
                article_notes=player.article_notes,
            )
        except Exception as exc:
            print(f"  profile search failed: {exc}")
        resolved.append(player)
        if delay > 0 and index < len(players):
            time.sleep(delay)
    return resolved


def visible_text_from_html(html_text: str) -> str:
    if BeautifulSoup is None:
        text = re.sub(r"<(script|style|noscript)\b.*?</\1>", " ", html_text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r"<[^>]+>", " ", text)
        return clean_text(text) or ""

    soup = BeautifulSoup(html_text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return clean_text(soup.get_text(" ")) or ""


def flatten_json_strings(value: Any) -> list[str]:
    strings: list[str] = []
    if isinstance(value, dict):
        for child in value.values():
            strings.extend(flatten_json_strings(child))
    elif isinstance(value, list):
        for child in value:
            strings.extend(flatten_json_strings(child))
    elif isinstance(value, str):
        strings.append(value)
    elif value is not None:
        strings.append(str(value))
    return strings


def embedded_json_text(html_text: str) -> str:
    if BeautifulSoup is None:
        chunks: list[str] = []
        for match in re.finditer(r"<script\b[^>]*>(.*?)</script>", html_text, flags=re.IGNORECASE | re.DOTALL):
            text = html.unescape(match.group(1))
            if "{" in text:
                chunks.append(text)
        return clean_text(" ".join(chunks)) or ""

    soup = BeautifulSoup(html_text, "html.parser")
    chunks: list[str] = []
    for script in soup.find_all("script"):
        text = script.string or script.get_text()
        if not text or "{" not in text:
            continue
        if script.get("id") == "__NEXT_DATA__":
            try:
                chunks.extend(flatten_json_strings(json.loads(text)))
                continue
            except json.JSONDecodeError:
                pass
        chunks.append(text)
    return clean_text(" ".join(chunks)) or ""


def regex_value(text: str, patterns: list[str]) -> str | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return clean_text(match.group(1))
    return None


def extract_basic_profile_fields(text: str) -> dict[str, str | None]:
    return {
        "height": regex_value(
            text,
            [
                r"\bHeight\s*[:\-]?\s*([0-7]\s*['-]\s*\d{1,2}\"?)",
                r"\b([0-7]\s*['-]\s*\d{1,2}\"?)\s*,?\s*\d{2,3}\s*(?:lb|lbs|pounds?|#)",
            ],
        ),
        "weight": regex_value(
            text,
            [
                r"\bWeight\s*[:\-]?\s*(\d{2,3}(?:\.\d+)?)\s*(?:lb|lbs|pounds?|#)?",
                r"\b[0-7]\s*['-]\s*\d{1,2}\"?\s*,?\s*(\d{2,3}(?:\.\d+)?)\s*(?:lb|lbs|pounds?|#)",
            ],
        ),
        "age": regex_value(text, [r"\bAge\s*[:\-]?\s*(\d{1,2}(?:\.\d+)?)"]),
        "birth_date": regex_value(
            text,
            [
                r"\bDOB\s*[:\-]?\s*([0-1]?\d[/-][0-3]?\d[/-]\d{2,4})",
                r"\bBirth Date\s*[:\-]?\s*([0-1]?\d[/-][0-3]?\d[/-]\d{2,4})",
            ],
        ),
        "bats": regex_value(text, [r"\bBats\s*[:\-]?\s*([RLS])\b", r"\bB/T\s*[:\-]?\s*([RLS])\s*/"]),
        "throws": regex_value(text, [r"\bThrows\s*[:\-]?\s*([RLS])\b", r"\bB/T\s*[:\-]?\s*[RLS]\s*/\s*([RLS])"]),
        "high_school": regex_value(text, [r"\bHigh School\s*[:\-]?\s*([A-Za-z0-9 .,'&()/-]+?)(?:\s{2,}| Grad| Commitment| Position|$)"]),
        "commitment": regex_value(text, [r"\bCommitment\s*[:\-]?\s*([A-Za-z0-9 .,'&()/-]+?)(?:\s{2,}| Grad| Position|$)"]),
        "position": regex_value(text, [r"\bPosition\s*[:\-]?\s*([A-Z0-9/,\- ]{1,20})(?:\s{2,}| B/T| Height|$)"]),
        "grad_year": regex_value(text, [r"\b(?:Class|Grad(?:uation)? Year)\s*[:\-]?\s*(20\d{2})"]),
    }


def metric_pattern(label: str) -> re.Pattern[str]:
    escaped = re.escape(label).replace(r"\ ", r"\s+")
    return re.compile(
        rf"\b{escaped}\b\s*[:\-]?\s*"
        rf"(\d{{1,3}}(?:\.\d+)?(?:\s*-\s*\d{{1,3}}(?:\.\d+)?)?)"
        rf"\s*(mph|sec|seconds?|rpm|g|ft|feet|in|inches)?",
        flags=re.IGNORECASE,
    )


def extract_metrics(text: str) -> dict[str, str | None]:
    metrics: dict[str, str | None] = {}
    for column, labels in METRIC_LABELS.items():
        value = None
        for label in labels:
            match = metric_pattern(label).search(text)
            if match:
                unit = match.group(2) or ""
                value = clean_text(f"{match.group(1)} {unit}")
                break
        metrics[column] = value
    return metrics


def update_non_empty(row: dict[str, Any], values: dict[str, Any], overwrite: bool = True) -> None:
    for key, value in values.items():
        if value in (None, ""):
            continue
        if overwrite or row.get(key) in (None, ""):
            row[key] = value


DYNAMIC_STAT_MAP = {
    "60 TIME": "sixty_time",
    "INF VELO": "infield_velocity",
    "OF VELO": "outfield_velocity",
    "C VELO": "catcher_velocity",
    "EXIT VELO": "max_exit_velocity",
    "HAND SPEED": "max_hand_speed",
    "BAT SPEED": "max_bat_speed",
    "FB MAX": "fastball_velocity",
    "FB SPIN RATE": "fastball_spin_rate",
    "CB": "curveball_velocity",
    "SL": "slider_velocity",
    "CH": "changeup_velocity",
}


def extract_dynamic_stat_boxes(html_text: str) -> dict[str, str]:
    metrics: dict[str, str] = {}
    box_pattern = re.compile(
        r'<div\b[^>]*class=["\'][^"\']*dynamic-stat-box[^"\']*stat[^"\']*["\'][^>]*>(.*?)</div>',
        flags=re.IGNORECASE | re.DOTALL,
    )
    value_pattern = re.compile(r"<strong\b[^>]*>(.*?)</strong>", flags=re.IGNORECASE | re.DOTALL)
    label_pattern = re.compile(r"<span\b[^>]*>(.*?)</span>", flags=re.IGNORECASE | re.DOTALL)
    for box_html in box_pattern.findall(html_text):
        value_match = value_pattern.search(box_html)
        label_match = label_pattern.search(box_html)
        if not value_match or not label_match:
            continue
        label = clean_text(re.sub(r"<br\s*/?>", " ", label_match.group(1), flags=re.IGNORECASE))
        value = clean_text(re.sub(r"<[^>]+>", " ", value_match.group(1)))
        column = DYNAMIC_STAT_MAP.get((label or "").upper())
        if column and value:
            metrics[column] = value
    return metrics


def extract_profile_header_fields(html_text: str) -> dict[str, str | None]:
    header_match = re.search(r'<div class="data-list">(.*?)</div>', html_text, flags=re.IGNORECASE | re.DOTALL)
    header_text = clean_text(re.sub(r"<br\s*/?>", " | ", header_match.group(1), flags=re.IGNORECASE)) if header_match else None
    header_text = clean_text(re.sub(r"<[^>]+>", " ", header_text or ""))
    fields = extract_basic_profile_fields(header_text or "")
    if header_text:
        parts = [clean_text(part) for part in header_text.split("|")]
        parts = [part for part in parts if part]
        if parts:
            fields["high_school"] = fields.get("high_school") or parts[0].replace(" • ", " / ")
        ht_wt_match = re.search(r"([0-7]'\s*\d{1,2}\"?)\s*[•\-\s]+(\d{2,3})\s*LBS", header_text, flags=re.IGNORECASE)
        if ht_wt_match:
            fields["height"] = clean_text(ht_wt_match.group(1))
            fields["weight"] = clean_text(ht_wt_match.group(2))
        bt_age_match = re.search(r"\b([RLS])\s*/\s*([RLS])\s*[•\-\s]+(\d+yr\s+\d+mo)", header_text)
        if bt_age_match:
            fields["bats"] = bt_age_match.group(1)
            fields["throws"] = bt_age_match.group(2)
            fields["age"] = bt_age_match.group(3)
        travel_match = re.search(r"Travel Team:\s*([^|]+)", header_text, flags=re.IGNORECASE)
        if travel_match:
            fields["travel_team"] = clean_text(travel_match.group(1))
    class_match = re.search(r"<h2>\s*CLASS OF\s+(20\d{2})\s*</h2>", html_text, flags=re.IGNORECASE)
    if class_match:
        fields["grad_year"] = class_match.group(1)
    return fields


def extract_profile_header_fields(html_text: str) -> dict[str, str | None]:
    header_match = re.search(r'<div class="data-list">(.*?)</div>', html_text, flags=re.IGNORECASE | re.DOTALL)
    header_html = header_match.group(1) if header_match else ""
    header_html = re.sub(r"<img\b[^>]*>", " ", header_html, flags=re.IGNORECASE)
    header_html = re.sub(r"<br\s*/?>", "\n", header_html, flags=re.IGNORECASE)
    header_text = clean_text(re.sub(r"<[^>]+>", " ", html.unescape(header_html))) or ""
    header_text = header_text.replace("&bullet;", "|").replace(chr(8226), "|").replace("â€¢", "|")

    fields = extract_basic_profile_fields(header_text)
    if header_text:
        parts = [clean_text(part) for part in re.split(r"\||\n", header_text)]
        parts = [part for part in parts if part]
        if parts:
            fields["high_school"] = fields.get("high_school") or parts[0]

        ht_wt_match = re.search(
            r"([0-7]\s*['-]\s*\d{1,2}\"?)\s*(?:\||-|,|\s)+\s*(\d{2,3}(?:\.\d+)?)\s*LBS",
            header_text,
            flags=re.IGNORECASE,
        )
        if ht_wt_match:
            fields["height"] = clean_text(ht_wt_match.group(1))
            fields["weight"] = clean_text(ht_wt_match.group(2))

        bt_age_match = re.search(r"\b([RLS])\s*/\s*([RLS])\s*(?:\||-|,|\s)+\s*(\d+yr\s+\d+mo)", header_text)
        if bt_age_match:
            fields["bats"] = bt_age_match.group(1)
            fields["throws"] = bt_age_match.group(2)
            fields["age"] = bt_age_match.group(3)

        travel_match = re.search(r"Travel Team:\s*([^|]+)", header_text, flags=re.IGNORECASE)
        if travel_match:
            fields["travel_team"] = clean_text(travel_match.group(1))

    class_match = re.search(r"<h2>\s*CLASS OF\s+(20\d{2})\s*</h2>", html_text, flags=re.IGNORECASE)
    if class_match:
        fields["grad_year"] = class_match.group(1)
    return fields


def extract_profile_name(html_text: str, profile_url: str) -> str | None:
    if BeautifulSoup is None:
        patterns = [
            r"<meta\b[^>]*property=[\"']og:title[\"'][^>]*content=[\"']([^\"']+)[\"']",
            r"<h1\b[^>]*>(.*?)</h1>",
            r"<title\b[^>]*>(.*?)</title>",
        ]
        for pattern in patterns:
            match = re.search(pattern, html_text, flags=re.IGNORECASE | re.DOTALL)
            if not match:
                continue
            value = re.sub(r"<[^>]+>", " ", html.unescape(match.group(1)))
            value = re.sub(r"\s*\|\s*Prep Baseball.*$", "", value, flags=re.IGNORECASE)
            value = clean_text(value)
            if value:
                return value
        return slug_to_name(profile_url)

    soup = BeautifulSoup(html_text, "html.parser")
    for selector in ["h1", "meta[property='og:title']", "title"]:
        element = soup.select_one(selector)
        if not element:
            continue
        value = element.get("content") if element.name == "meta" else element.get_text(" ")
        value = re.sub(r"\s*\|\s*Prep Baseball.*$", "", value or "", flags=re.IGNORECASE)
        value = clean_text(value)
        if value:
            return value
    return slug_to_name(profile_url)


def parse_profile(html_text: str, player: PlayerLink) -> dict[str, Any]:
    visible_text = visible_text_from_html(html_text)
    json_text = embedded_json_text(html_text)
    combined_text = clean_text(f"{visible_text} {json_text}") or ""

    profile_url = player.profile_url or ""
    parsed_url = urlparse(profile_url)
    path_parts = [part for part in parsed_url.path.split("/") if part]
    profile_state = path_parts[1] if len(path_parts) > 2 and path_parts[0] == "profiles" else None

    row: dict[str, Any] = {
        "rank": player.rank,
        "player_name": player.player_name or extract_profile_name(html_text, profile_url),
        "profile_name": extract_profile_name(html_text, profile_url),
        "profile_url": player.profile_url,
        "profile_state": profile_state,
        "article_position": player.article_position,
        "article_school": player.article_school,
        "article_state": player.article_state,
        "article_notes": player.article_notes,
    }
    update_non_empty(row, extract_basic_profile_fields(combined_text))
    update_non_empty(row, extract_profile_header_fields(html_text))
    update_non_empty(row, extract_metrics(combined_text))
    update_non_empty(row, extract_metrics(player.article_notes or ""), overwrite=False)
    update_non_empty(row, extract_dynamic_stat_boxes(html_text))
    return row


def scrape_players(players: list[PlayerLink], session: requests.Session, delay: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, player in enumerate(players, start=1):
        print(f"[{index}/{len(players)}] {player.player_name or player.profile_url}")
        if not player.profile_url:
            row = {
                "rank": player.rank,
                "player_name": player.player_name,
                "profile_url": None,
                "article_position": player.article_position,
                "article_school": player.article_school,
                "article_state": player.article_state,
                "article_notes": player.article_notes,
                "scrape_error": "No profile URL resolved.",
            }
            update_non_empty(row, extract_metrics(player.article_notes or ""))
            rows.append(row)
            continue
        try:
            profile_html = fetch_html(player.profile_url, session)
            rows.append(parse_profile(profile_html, player))
        except Exception as exc:
            rows.append(
                {
                    "rank": player.rank,
                    "player_name": player.player_name,
                    "profile_url": player.profile_url,
                    "scrape_error": str(exc),
                }
            )
        if delay > 0 and index < len(players):
            time.sleep(delay)
    return pd.DataFrame(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scrape PBR draft-board profile metrics.")
    parser.add_argument("--draft-board-url", default=DEFAULT_DRAFT_BOARD_URL)
    parser.add_argument("--article-html", type=Path, help="Saved draft-board HTML to parse instead of fetching live.")
    parser.add_argument("--input-csv", type=Path, help="CSV with profile_url plus optional rank/name/position/school/state.")
    parser.add_argument("--outdir", type=Path, default=Path("data/pbr"))
    parser.add_argument("--output-name", default="pbr_2026_draft_board_metrics.csv")
    parser.add_argument("--class-year", default="2026", help="PBR class year used when resolving profile-search results.")
    parser.add_argument("--no-resolve-profiles", action="store_true", help="Skip PBR profile-search lookups for table rows without links.")
    parser.add_argument("--limit", type=int, help="Only scrape the first N discovered players.")
    parser.add_argument("--delay", type=float, default=0.75, help="Seconds to wait between profile requests.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    session = build_session()

    if args.input_csv:
        players = load_player_links_from_csv(args.input_csv)
    else:
        article_html = args.article_html.read_text(encoding="utf-8") if args.article_html else fetch_html(args.draft_board_url, session)
        players = parse_draft_board(article_html, args.draft_board_url)

    if args.limit:
        players = players[: args.limit]

    if not players:
        raise RuntimeError("No players to scrape.")

    if not args.no_resolve_profiles:
        players = resolve_missing_profile_urls(players, session, args.draft_board_url, args.class_year, args.delay)

    frame = scrape_players(players, session, args.delay)
    args.outdir.mkdir(parents=True, exist_ok=True)
    output_path = args.outdir / args.output_name
    frame.to_csv(output_path, index=False)
    print(f"Saved {len(frame):,} player rows to: {output_path}")


if __name__ == "__main__":
    main()
