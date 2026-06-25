import json
import math
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

try:
    from scotland_r32_live_sim import GroupStandingRow
except ImportError:
    class GroupStandingRow(NamedTuple):
        Group: str
        Pos: int
        Team: str
        Pts: int
        GD: int
        GF: int
        GA: int
        W: int
        D: int
        L: int
        P: int

import numpy as np
import pandas as pd
import requests
import streamlit as st
from scipy.stats import poisson


st.set_page_config(page_title="FIFA World Cup 2026 Live Monte Carlo", layout="wide")


GROUPS: Dict[str, List[str]] = {
    "A": ["Mexico", "South Africa", "Korea Republic", "Czechia"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Haiti", "Scotland", "Brazil", "Morocco"],
    "D": ["USA", "Paraguay", "Australia", "Turkiye"],
    "E": ["Cote d'Ivoire", "Ecuador", "Germany", "Curacao"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["IR Iran", "New Zealand", "Belgium", "Egypt"],
    "H": ["Saudi Arabia", "Uruguay", "Spain", "Cabo Verde"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "Congo DR", "Uzbekistan", "Colombia"],
    "L": ["Ghana", "Panama", "England", "Croatia"],
}

GROUP_FIXTURES: Dict[str, List[Tuple[str, str]]] = {
    "A": [
        ("Mexico", "South Africa"),
        ("Korea Republic", "Czechia"),
        ("Czechia", "South Africa"),
        ("Mexico", "Korea Republic"),
        ("Czechia", "Mexico"),
        ("South Africa", "Korea Republic"),
    ],
    "B": [
        ("Canada", "Bosnia and Herzegovina"),
        ("Qatar", "Switzerland"),
        ("Switzerland", "Bosnia and Herzegovina"),
        ("Canada", "Qatar"),
        ("Switzerland", "Canada"),
        ("Bosnia and Herzegovina", "Qatar"),
    ],
    "C": [
        ("Haiti", "Scotland"),
        ("Brazil", "Morocco"),
        ("Brazil", "Haiti"),
        ("Scotland", "Morocco"),
        ("Scotland", "Brazil"),
        ("Morocco", "Haiti"),
    ],
    "D": [
        ("USA", "Paraguay"),
        ("Australia", "Turkiye"),
        ("Turkiye", "Paraguay"),
        ("USA", "Australia"),
        ("Turkiye", "USA"),
        ("Paraguay", "Australia"),
    ],
    "E": [
        ("Cote d'Ivoire", "Ecuador"),
        ("Germany", "Curacao"),
        ("Germany", "Cote d'Ivoire"),
        ("Ecuador", "Curacao"),
        ("Curacao", "Cote d'Ivoire"),
        ("Ecuador", "Germany"),
    ],
    "F": [
        ("Netherlands", "Japan"),
        ("Sweden", "Tunisia"),
        ("Netherlands", "Sweden"),
        ("Tunisia", "Japan"),
        ("Japan", "Sweden"),
        ("Tunisia", "Netherlands"),
    ],
    "G": [
        ("IR Iran", "New Zealand"),
        ("Belgium", "Egypt"),
        ("Belgium", "IR Iran"),
        ("New Zealand", "Egypt"),
        ("Egypt", "IR Iran"),
        ("New Zealand", "Belgium"),
    ],
    "H": [
        ("Saudi Arabia", "Uruguay"),
        ("Spain", "Cabo Verde"),
        ("Uruguay", "Cabo Verde"),
        ("Spain", "Saudi Arabia"),
        ("Cabo Verde", "Saudi Arabia"),
        ("Uruguay", "Spain"),
    ],
    "I": [
        ("France", "Senegal"),
        ("Iraq", "Norway"),
        ("Norway", "Senegal"),
        ("France", "Iraq"),
        ("Norway", "France"),
        ("Senegal", "Iraq"),
    ],
    "J": [
        ("Argentina", "Algeria"),
        ("Austria", "Jordan"),
        ("Argentina", "Austria"),
        ("Jordan", "Algeria"),
        ("Algeria", "Austria"),
        ("Jordan", "Argentina"),
    ],
    "K": [
        ("Portugal", "Congo DR"),
        ("Uzbekistan", "Colombia"),
        ("Portugal", "Uzbekistan"),
        ("Colombia", "Congo DR"),
        ("Colombia", "Portugal"),
        ("Congo DR", "Uzbekistan"),
    ],
    "L": [
        ("Ghana", "Panama"),
        ("England", "Croatia"),
        ("England", "Ghana"),
        ("Panama", "Croatia"),
        ("Panama", "England"),
        ("Croatia", "Ghana"),
    ],
}

# Baseline completed matches through 19 June 2026 (locked if ESPN has no newer value).
BASELINE_COMPLETED_MATCHES: List[Tuple[str, str, str, int, int]] = [
    ("A", "Mexico", "South Africa", 2, 0),
    ("A", "Korea Republic", "Czechia", 2, 1),
    ("A", "Czechia", "South Africa", 1, 1),
    ("A", "Mexico", "Korea Republic", 1, 0),
    ("B", "Canada", "Bosnia and Herzegovina", 1, 1),
    ("B", "Qatar", "Switzerland", 1, 1),
    ("B", "Switzerland", "Bosnia and Herzegovina", 4, 1),
    ("B", "Canada", "Qatar", 6, 0),
    ("C", "Haiti", "Scotland", 0, 1),
    ("C", "Brazil", "Morocco", 1, 1),
    ("C", "Brazil", "Haiti", 3, 0),
    ("C", "Scotland", "Morocco", 0, 1),
    ("D", "USA", "Paraguay", 4, 1),
    ("D", "Australia", "Turkiye", 2, 0),
    ("D", "Turkiye", "Paraguay", 0, 1),
    ("D", "USA", "Australia", 2, 0),
    ("E", "Cote d'Ivoire", "Ecuador", 1, 0),
    ("E", "Germany", "Curacao", 7, 1),
    ("F", "Netherlands", "Japan", 2, 2),
    ("F", "Sweden", "Tunisia", 5, 1),
    ("G", "IR Iran", "New Zealand", 2, 2),
    ("G", "Belgium", "Egypt", 1, 1),
    ("H", "Saudi Arabia", "Uruguay", 1, 1),
    ("H", "Spain", "Cabo Verde", 0, 0),
    ("I", "France", "Senegal", 3, 1),
    ("I", "Iraq", "Norway", 1, 4),
    ("J", "Argentina", "Algeria", 3, 0),
    ("J", "Austria", "Jordan", 3, 1),
    ("K", "Portugal", "Congo DR", 1, 1),
    ("K", "Uzbekistan", "Colombia", 1, 3),
    ("L", "Ghana", "Panama", 1, 0),
    ("L", "England", "Croatia", 4, 2),
]

FIFA_RANK: Dict[str, int] = {
    "Argentina": 1,
    "France": 2,
    "Spain": 3,
    "England": 4,
    "Brazil": 5,
    "Belgium": 6,
    "Portugal": 7,
    "Netherlands": 8,
    "Germany": 9,
    "Croatia": 10,
    "Uruguay": 11,
    "Colombia": 12,
    "Morocco": 13,
    "Mexico": 14,
    "USA": 15,
    "Switzerland": 16,
    "Japan": 17,
    "Senegal": 18,
    "IR Iran": 19,
    "Austria": 20,
    "Korea Republic": 21,
    "Ecuador": 22,
    "Sweden": 23,
    "Paraguay": 24,
    "Norway": 25,
    "Czechia": 26,
    "Egypt": 27,
    "Australia": 28,
    "Scotland": 29,
    "Cote d'Ivoire": 30,
    "Algeria": 31,
    "Canada": 32,
    "Tunisia": 33,
    "Iraq": 34,
    "Ghana": 35,
    "South Africa": 36,
    "Panama": 37,
    "Bosnia and Herzegovina": 38,
    "Uzbekistan": 39,
    "Jordan": 40,
    "Cabo Verde": 41,
    "Qatar": 42,
    "New Zealand": 43,
    "Congo DR": 44,
    "Curacao": 45,
    "Saudi Arabia": 46,
    "Haiti": 47,
    "Turkiye": 48,
}

XG_OVERRIDES: Dict[str, Tuple[float, float]] = {
    "Argentina": (2.05, 0.78),
    "France": (2.00, 0.82),
    "Spain": (1.88, 0.86),
    "England": (1.95, 0.90),
    "Brazil": (1.98, 0.84),
    "Germany": (1.87, 0.95),
    "Netherlands": (1.82, 0.92),
    "Portugal": (1.84, 0.93),
    "Belgium": (1.76, 1.00),
    "USA": (1.62, 1.06),
    "Mexico": (1.58, 1.07),
    "Morocco": (1.55, 0.98),
    "Japan": (1.54, 1.02),
    "Croatia": (1.52, 1.04),
    "Uruguay": (1.61, 1.00),
    "Colombia": (1.60, 1.01),
}

APPENDIX_C_SLOT_ORDER = ("M74", "M77", "M79", "M80", "M81", "M82", "M85", "M87")


def _load_appendix_c_matrix() -> Dict[str, Dict[str, str]]:
    matrix_path = Path(__file__).with_name("appendix_c_matrix.json")
    with matrix_path.open("r", encoding="utf-8") as f:
        matrix = json.load(f)
    return matrix


APPENDIX_C_MATRIX = _load_appendix_c_matrix()


def normalize_name(name: str) -> str:
    normalized = unicodedata.normalize("NFKD", name)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    lowered = ascii_only.lower().replace("-", " ").replace("'", " ").replace(".", " ")
    compact = " ".join(lowered.split())
    return compact


CANONICAL_TEAMS = sorted({team for teams in GROUPS.values() for team in teams})
CANONICAL_BY_NORMALIZED = {normalize_name(t): t for t in CANONICAL_TEAMS}
TEAM_ALIAS = {
    "south korea": "Korea Republic",
    "korea republic": "Korea Republic",
    "united states": "USA",
    "usa": "USA",
    "iran": "IR Iran",
    "ir iran": "IR Iran",
    "cape verde": "Cabo Verde",
    "cabo verde": "Cabo Verde",
    "dr congo": "Congo DR",
    "congo dr": "Congo DR",
    "congo democratic republic": "Congo DR",
    "ivory coast": "Cote d'Ivoire",
    "cote divoire": "Cote d'Ivoire",
    "turkiye": "Turkiye",
    "turkey": "Turkiye",
    "bosnia herzegovina": "Bosnia and Herzegovina",
    "bosnia and herzegovina": "Bosnia and Herzegovina",
}

ALL_TEAMS = CANONICAL_TEAMS
TEAM_INDEX = {team: i for i, team in enumerate(ALL_TEAMS)}
TEAM_GROUP = {team: g for g, teams in GROUPS.items() for team in teams}
GROUP_ORDER = list(GROUPS.keys())

STAT_P = 0
STAT_W = 1
STAT_D = 2
STAT_L = 3
STAT_GF = 4
STAT_GA = 5
STAT_PTS = 6

GROUP_TEAM_INDICES = {group: np.array([TEAM_INDEX[t] for t in teams], dtype=np.int16) for group, teams in GROUPS.items()}
GROUP_FIXTURE_INDICES = {
    group: [(TEAM_INDEX[h], TEAM_INDEX[a]) for h, a in fixtures] for group, fixtures in GROUP_FIXTURES.items()
}

FIXTURE_ORIENTATION: Dict[str, Dict[frozenset, Tuple[str, str]]] = {}
for group, fixtures in GROUP_FIXTURES.items():
    FIXTURE_ORIENTATION[group] = {frozenset((h, a)): (h, a) for h, a in fixtures}


def resolve_team_name(raw_name: str) -> Optional[str]:
    key = normalize_name(raw_name)
    if key in TEAM_ALIAS:
        return TEAM_ALIAS[key]
    return CANONICAL_BY_NORMALIZED.get(key)


def infer_group(team_a: str, team_b: str) -> Optional[str]:
    ga = TEAM_GROUP.get(team_a)
    gb = TEAM_GROUP.get(team_b)
    if ga is not None and ga == gb:
        return ga
    return None


@st.cache_data(ttl=120)
def fetch_espn_scoreboard() -> Dict:
    url = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard"
    params = {"dates": "20260611-20260719"}
    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()
    return response.json()


def parse_espn_matches(payload: Dict) -> Tuple[List[Tuple[str, str, str, int, int]], pd.DataFrame, List[str]]:
    completed = []
    live_rows = []
    warnings = []

    events = payload.get("events", [])

    for event in events:
        competitions = event.get("competitions", [])
        if not competitions:
            continue

        comp = competitions[0]
        status_type = comp.get("status", {}).get("type", {})
        state = status_type.get("state", "")
        description = status_type.get("description", "")
        is_completed = bool(status_type.get("completed", False))

        competitors = comp.get("competitors", [])
        if len(competitors) < 2:
            continue

        home_obj = next((c for c in competitors if c.get("homeAway") == "home"), competitors[0])
        away_obj = next((c for c in competitors if c.get("homeAway") == "away"), competitors[1])

        home_name_raw = home_obj.get("team", {}).get("displayName", "")
        away_name_raw = away_obj.get("team", {}).get("displayName", "")

        home_team = resolve_team_name(home_name_raw)
        away_team = resolve_team_name(away_name_raw)
        if not home_team or not away_team:
            warnings.append(f"Unmapped team name: {home_name_raw} vs {away_name_raw}")
            continue

        group = infer_group(home_team, away_team)
        if group is None:
            warnings.append(f"Could not infer group for match: {home_team} vs {away_team}")
            continue

        home_score = int(float(home_obj.get("score", 0) or 0))
        away_score = int(float(away_obj.get("score", 0) or 0))

        if is_completed:
            completed.append((group, home_team, away_team, home_score, away_score))
        elif state in {"in", "halftime", "delay", "suspended"}:
            live_rows.append(
                {
                    "Group": group,
                    "Home": home_team,
                    "Away": away_team,
                    "Score": f"{home_score}-{away_score}",
                    "Status": description,
                    "Clock": comp.get("status", {}).get("displayClock", ""),
                }
            )

    live_df = pd.DataFrame(live_rows)
    return completed, live_df, sorted(set(warnings))


def build_strength_table() -> pd.DataFrame:
    rows = []
    for team in ALL_TEAMS:
        rank = FIFA_RANK[team]
        elo = 1975 - 7.0 * rank
        if team in XG_OVERRIDES:
            xg_for, xg_against = XG_OVERRIDES[team]
        else:
            xg_for = float(np.clip(1.00 + (elo - 1550) / 430, 0.80, 1.80))
            xg_against = float(np.clip(1.62 - (elo - 1550) / 520, 0.90, 1.80))
        rows.append({"team": team, "rank": rank, "elo": elo, "xg_for": xg_for, "xg_against": xg_against})
    return pd.DataFrame(rows).set_index("team")


def expected_goals(home: str, away: str, strength: pd.DataFrame) -> Tuple[float, float]:
    s_h = strength.loc[home]
    s_a = strength.loc[away]

    elo_term = (s_h["elo"] - s_a["elo"]) / 400.0
    rank_term = (s_a["rank"] - s_h["rank"]) / 60.0

    xg_attack_h = s_h["xg_for"] / strength["xg_for"].mean()
    xg_def_opp_h = s_a["xg_against"] / strength["xg_against"].mean()
    xg_attack_a = s_a["xg_for"] / strength["xg_for"].mean()
    xg_def_opp_a = s_h["xg_against"] / strength["xg_against"].mean()

    base = 1.33
    lam_h = base * math.exp(0.47 * elo_term + 0.22 * rank_term) * xg_attack_h * xg_def_opp_h
    lam_a = base * math.exp(-0.47 * elo_term - 0.22 * rank_term) * xg_attack_a * xg_def_opp_a
    return float(np.clip(lam_h, 0.15, 3.8)), float(np.clip(lam_a, 0.15, 3.8))


def _unused_distribution_anchor(mu: float) -> float:
    # Keeps scipy.stats.poisson explicitly in the model stack while simulation uses fast NumPy draws.
    return float(poisson.pmf(0, mu))


def _precompute_goal_rate_matrices(strength: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rank = strength.loc[ALL_TEAMS, "rank"].to_numpy(dtype=np.float64)
    elo = strength.loc[ALL_TEAMS, "elo"].to_numpy(dtype=np.float64)
    xg_for = strength.loc[ALL_TEAMS, "xg_for"].to_numpy(dtype=np.float64)
    xg_against = strength.loc[ALL_TEAMS, "xg_against"].to_numpy(dtype=np.float64)

    elo_term = (elo[:, None] - elo[None, :]) / 400.0
    rank_term = (rank[None, :] - rank[:, None]) / 60.0

    xg_for_mean = xg_for.mean()
    xg_against_mean = xg_against.mean()

    xg_attack_h = xg_for[:, None] / xg_for_mean
    xg_def_opp_h = xg_against[None, :] / xg_against_mean
    xg_attack_a = xg_for[None, :] / xg_for_mean
    xg_def_opp_a = xg_against[:, None] / xg_against_mean

    base = 1.33
    lam_home = base * np.exp(0.47 * elo_term + 0.22 * rank_term) * xg_attack_h * xg_def_opp_h
    lam_away = base * np.exp(-0.47 * elo_term - 0.22 * rank_term) * xg_attack_a * xg_def_opp_a

    lam_home = np.clip(lam_home, 0.15, 3.8)
    lam_away = np.clip(lam_away, 0.15, 3.8)

    return lam_home, lam_away, elo


def _apply_result_array(stats: np.ndarray, home_idx: int, away_idx: int, gh: int, ga: int) -> None:
    stats[home_idx, STAT_P] += 1
    stats[away_idx, STAT_P] += 1

    stats[home_idx, STAT_GF] += gh
    stats[home_idx, STAT_GA] += ga
    stats[away_idx, STAT_GF] += ga
    stats[away_idx, STAT_GA] += gh

    if gh > ga:
        stats[home_idx, STAT_W] += 1
        stats[away_idx, STAT_L] += 1
        stats[home_idx, STAT_PTS] += 3
    elif gh < ga:
        stats[away_idx, STAT_W] += 1
        stats[home_idx, STAT_L] += 1
        stats[away_idx, STAT_PTS] += 3
    else:
        stats[home_idx, STAT_D] += 1
        stats[away_idx, STAT_D] += 1
        stats[home_idx, STAT_PTS] += 1
        stats[away_idx, STAT_PTS] += 1


def _sorted_group_indices(stats: np.ndarray, team_indices: np.ndarray, elo_arr: np.ndarray) -> np.ndarray:
    pts = stats[team_indices, STAT_PTS]
    gd = stats[team_indices, STAT_GF] - stats[team_indices, STAT_GA]
    gf = stats[team_indices, STAT_GF]
    elo = elo_arr[team_indices]

    order = np.lexsort((-elo, -gf, -gd, -pts))
    return team_indices[order]


def _build_group_table_from_stats(stats: np.ndarray, elo_arr: np.ndarray) -> pd.DataFrame:
    rows = []
    for group in GROUP_ORDER:
        sorted_indices = _sorted_group_indices(stats, GROUP_TEAM_INDICES[group], elo_arr)
        for pos, team_idx in enumerate(sorted_indices, start=1):
            gf = int(stats[team_idx, STAT_GF])
            ga = int(stats[team_idx, STAT_GA])
            rows.append(
                {
                    "Group": group,
                    "Pos": pos,
                    "Team": ALL_TEAMS[team_idx],
                    "Pts": int(stats[team_idx, STAT_PTS]),
                    "GD": gf - ga,
                    "GF": gf,
                    "GA": ga,
                    "W": int(stats[team_idx, STAT_W]),
                    "D": int(stats[team_idx, STAT_D]),
                    "L": int(stats[team_idx, STAT_L]),
                    "P": int(stats[team_idx, STAT_P]),
                }
            )
    return pd.DataFrame(rows)


def _assign_third_place_slots_idx(best_third_groups: List[str], best_third_team_indices: List[int]) -> Dict[str, int]:
    group_to_team_idx = dict(zip(best_third_groups, best_third_team_indices))
    key = "".join(sorted(best_third_groups))
    assigned_groups = APPENDIX_C_MATRIX.get(key)
    if assigned_groups is None:
        raise RuntimeError(f"No Appendix C mapping for third-place combination: {key}")

    return {slot: group_to_team_idx[assigned_groups[slot]] for slot in APPENDIX_C_SLOT_ORDER}


def _knockout_match_idx(
    team_a_idx: int,
    team_b_idx: int,
    lam_home: np.ndarray,
    lam_away: np.ndarray,
    elo_arr: np.ndarray,
    rng: np.random.Generator,
) -> int:
    g_a = int(rng.poisson(lam_home[team_a_idx, team_b_idx]))
    g_b = int(rng.poisson(lam_away[team_a_idx, team_b_idx]))

    if g_a > g_b:
        return team_a_idx
    if g_b > g_a:
        return team_b_idx

    et_a = int(rng.poisson(lam_home[team_a_idx, team_b_idx] * 0.33))
    et_b = int(rng.poisson(lam_away[team_a_idx, team_b_idx] * 0.33))
    g_a += et_a
    g_b += et_b

    if g_a != g_b:
        return team_a_idx if g_a > g_b else team_b_idx

    elo_diff = elo_arr[team_a_idx] - elo_arr[team_b_idx]
    p_team_a = 1.0 / (1.0 + math.exp(-elo_diff / 115.0))
    return team_a_idx if rng.random() < p_team_a else team_b_idx


def merge_locked_results(espn_completed: List[Tuple[str, str, str, int, int]]) -> Dict[Tuple[str, str, str], Tuple[int, int]]:
    locked = {(g, h, a): (gh, ga) for g, h, a, gh, ga in BASELINE_COMPLETED_MATCHES}

    for group, t1, t2, g1, g2 in espn_completed:
        canonical = FIXTURE_ORIENTATION[group].get(frozenset((t1, t2)))
        if canonical is None:
            continue
        home, away = canonical
        if (t1, t2) == (home, away):
            locked[(group, home, away)] = (g1, g2)
        else:
            locked[(group, home, away)] = (g2, g1)

    return locked


def init_group_table(group: str) -> Dict[str, Dict[str, int]]:
    return {
        t: {"P": 0, "W": 0, "D": 0, "L": 0, "GF": 0, "GA": 0, "GD": 0, "Pts": 0}
        for t in GROUPS[group]
    }


def apply_result(table: Dict[str, Dict[str, int]], home: str, away: str, gh: int, ga: int) -> None:
    table[home]["P"] += 1
    table[away]["P"] += 1
    table[home]["GF"] += gh
    table[home]["GA"] += ga
    table[away]["GF"] += ga
    table[away]["GA"] += gh

    if gh > ga:
        table[home]["W"] += 1
        table[away]["L"] += 1
        table[home]["Pts"] += 3
    elif gh < ga:
        table[away]["W"] += 1
        table[home]["L"] += 1
        table[away]["Pts"] += 3
    else:
        table[home]["D"] += 1
        table[away]["D"] += 1
        table[home]["Pts"] += 1
        table[away]["Pts"] += 1

    table[home]["GD"] = table[home]["GF"] - table[home]["GA"]
    table[away]["GD"] = table[away]["GF"] - table[away]["GA"]


def sort_group(group: str, table: Dict[str, Dict[str, int]], strength: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for team, stats in table.items():
        row = {"Group": group, "Team": team}
        row.update(stats)
        row["Elo"] = float(strength.loc[team, "elo"])
        rows.append(row)
    df = pd.DataFrame(rows)
    df = df.sort_values(by=["Pts", "GD", "GF", "Elo"], ascending=[False, False, False, False]).reset_index(drop=True)
    df["Pos"] = df.index + 1
    return df[["Group", "Pos", "Team", "Pts", "GD", "GF", "GA", "W", "D", "L", "P"]]


def assign_third_place_slots(best_thirds: pd.DataFrame) -> Dict[str, str]:
    group_to_team = {row["Group"]: row["Team"] for _, row in best_thirds.iterrows()}
    key = "".join(sorted(group_to_team.keys()))
    assigned_groups = APPENDIX_C_MATRIX.get(key)
    if assigned_groups is None:
        raise RuntimeError(f"No Appendix C mapping for third-place combination: {key}")

    return {slot: group_to_team[assigned_groups[slot]] for slot in APPENDIX_C_SLOT_ORDER}


def knockout_match(team_a: str, team_b: str, strength: pd.DataFrame, rng: np.random.Generator) -> Tuple[str, str]:
    lam_a, lam_b = expected_goals(team_a, team_b, strength)
    g_a = int(rng.poisson(lam_a))
    g_b = int(rng.poisson(lam_b))

    if g_a > g_b:
        return team_a, f"{team_a} {g_a}-{g_b} {team_b}"
    if g_b > g_a:
        return team_b, f"{team_a} {g_a}-{g_b} {team_b}"

    et_a = int(rng.poisson(lam_a * 0.33))
    et_b = int(rng.poisson(lam_b * 0.33))
    g_a += et_a
    g_b += et_b

    if g_a != g_b:
        winner = team_a if g_a > g_b else team_b
        return winner, f"{team_a} {g_a}-{g_b} {team_b} (AET)"

    elo_diff = strength.loc[team_a, "elo"] - strength.loc[team_b, "elo"]
    p_team_a = 1.0 / (1.0 + math.exp(-elo_diff / 115.0))
    winner = team_a if rng.random() < p_team_a else team_b
    return winner, f"{team_a} {g_a}-{g_b} {team_b} (PEN {winner})"


def locked_group_tables(locked_lookup: Dict[Tuple[str, str, str], Tuple[int, int]], strength: pd.DataFrame) -> pd.DataFrame:
    group_tables = []
    for group in GROUPS:
        table = init_group_table(group)
        for home, away in GROUP_FIXTURES[group]:
            key = (group, home, away)
            if key in locked_lookup:
                gh, ga = locked_lookup[key]
                apply_result(table, home, away, gh, ga)
        group_tables.append(sort_group(group, table, strength))
    return pd.concat(group_tables, ignore_index=True)


@st.cache_data(show_spinner=False)
def run_monte_carlo(
    locked_matches: List[Tuple[str, str, str, int, int]],
    iterations: int,
    seed: int,
) -> Dict:
    rng = np.random.default_rng(seed)
    strength = build_strength_table()
    _unused_distribution_anchor(1.1)

    lam_home, lam_away, elo_arr = _precompute_goal_rate_matrices(strength)

    locked_lookup_idx = {
        (g, TEAM_INDEX[h], TEAM_INDEX[a]): (gh, ga) for g, h, a, gh, ga in locked_matches
    }

    advance_r32 = np.zeros(len(ALL_TEAMS), dtype=np.int64)
    reach_r16 = np.zeros(len(ALL_TEAMS), dtype=np.int64)
    reach_qf = np.zeros(len(ALL_TEAMS), dtype=np.int64)
    reach_sf = np.zeros(len(ALL_TEAMS), dtype=np.int64)
    reach_final = np.zeros(len(ALL_TEAMS), dtype=np.int64)
    win_tournament = np.zeros(len(ALL_TEAMS), dtype=np.int64)

    finish_top2 = np.zeros(len(ALL_TEAMS), dtype=np.int64)
    finish_best3 = np.zeros(len(ALL_TEAMS), dtype=np.int64)

    path_counters = {
        "R32": {m: Counter() for m in range(73, 89)},
        "R16": {m: Counter() for m in range(89, 97)},
        "QF": {m: Counter() for m in range(97, 101)},
        "SF": {m: Counter() for m in range(101, 103)},
        "3P": {103: Counter()},
        "Final": {104: Counter()},
    }

    latest_stats = np.zeros((len(ALL_TEAMS), 7), dtype=np.int16)
    # Frequency tables for modal (most-likely) final group stats.
    # pts: 0-9  (3 games × 3 pts max)
    # gd:  -12 to +12  stored at index gd+12  (25 slots)
    # gf:   0 to 19    (20 slots)
    pts_freq = np.zeros((len(ALL_TEAMS), 10), dtype=np.int32)
    gd_freq  = np.zeros((len(ALL_TEAMS), 25), dtype=np.int32)
    gf_freq  = np.zeros((len(ALL_TEAMS), 20), dtype=np.int32)

    for _ in range(iterations):
        stats = np.zeros((len(ALL_TEAMS), 7), dtype=np.int16)

        winners_idx: Dict[str, int] = {}
        runners_idx: Dict[str, int] = {}
        third_rows: List[Tuple[str, int, int, int, int]] = []

        for group in GROUP_ORDER:
            for home_idx, away_idx in GROUP_FIXTURE_INDICES[group]:
                key = (group, home_idx, away_idx)
                if key in locked_lookup_idx:
                    gh, ga = locked_lookup_idx[key]
                else:
                    gh = int(rng.poisson(lam_home[home_idx, away_idx]))
                    ga = int(rng.poisson(lam_away[home_idx, away_idx]))
                _apply_result_array(stats, home_idx, away_idx, gh, ga)

            sorted_group = _sorted_group_indices(stats, GROUP_TEAM_INDICES[group], elo_arr)
            winners_idx[group] = int(sorted_group[0])
            runners_idx[group] = int(sorted_group[1])
            third_idx = int(sorted_group[2])

            third_rows.append(
                (
                    group,
                    third_idx,
                    int(stats[third_idx, STAT_PTS]),
                    int(stats[third_idx, STAT_GF] - stats[third_idx, STAT_GA]),
                    int(stats[third_idx, STAT_GF]),
                )
            )

        # Vectorised frequency-table update for modal predicted standings.
        pts_vals = np.clip(stats[:, STAT_PTS].astype(np.int32), 0, 9)
        pts_freq[np.arange(len(ALL_TEAMS)), pts_vals] += 1
        gd_vals = np.clip((stats[:, STAT_GF] - stats[:, STAT_GA]).astype(np.int32), -12, 12)
        gd_freq[np.arange(len(ALL_TEAMS)), gd_vals + 12] += 1
        gf_vals = np.clip(stats[:, STAT_GF].astype(np.int32), 0, 19)
        gf_freq[np.arange(len(ALL_TEAMS)), gf_vals] += 1

        for team_idx in winners_idx.values():
            finish_top2[team_idx] += 1
        for team_idx in runners_idx.values():
            finish_top2[team_idx] += 1

        third_rows_sorted = sorted(third_rows, key=lambda x: (x[2], x[3], x[4]), reverse=True)
        best_thirds = third_rows_sorted[:8]

        best_third_groups = [row[0] for row in best_thirds]
        best_third_indices = [row[1] for row in best_thirds]
        for team_idx in best_third_indices:
            finish_best3[team_idx] += 1

        qualifiers = set(list(winners_idx.values()) + list(runners_idx.values()) + best_third_indices)
        for team_idx in qualifiers:
            advance_r32[team_idx] += 1

        third_slots_idx = _assign_third_place_slots_idx(best_third_groups, best_third_indices)

        r32 = {
            73: (runners_idx["A"], runners_idx["B"]),
            74: (winners_idx["E"], third_slots_idx["M74"]),
            75: (winners_idx["F"], runners_idx["C"]),
            76: (winners_idx["C"], runners_idx["F"]),
            77: (winners_idx["I"], third_slots_idx["M77"]),
            78: (runners_idx["E"], runners_idx["I"]),
            79: (winners_idx["A"], third_slots_idx["M79"]),
            80: (winners_idx["L"], third_slots_idx["M80"]),
            81: (winners_idx["D"], third_slots_idx["M81"]),
            82: (winners_idx["G"], third_slots_idx["M82"]),
            83: (runners_idx["K"], runners_idx["L"]),
            84: (winners_idx["H"], runners_idx["J"]),
            85: (winners_idx["B"], third_slots_idx["M85"]),
            86: (winners_idx["J"], runners_idx["H"]),
            87: (winners_idx["K"], third_slots_idx["M87"]),
            88: (runners_idx["D"], runners_idx["G"]),
        }

        winners_by_match = {}
        losers_by_match = {}

        for mid in range(73, 89):
            t1, t2 = r32[mid]
            winner = _knockout_match_idx(t1, t2, lam_home, lam_away, elo_arr, rng)
            winners_by_match[mid] = winner
            losers_by_match[mid] = t2 if winner == t1 else t1
            path_counters["R32"][mid][(t1, t2, winner)] += 1
            reach_r16[winner] += 1

        r16 = {
            89: (winners_by_match[74], winners_by_match[77]),
            90: (winners_by_match[73], winners_by_match[75]),
            91: (winners_by_match[76], winners_by_match[78]),
            92: (winners_by_match[79], winners_by_match[80]),
            93: (winners_by_match[83], winners_by_match[84]),
            94: (winners_by_match[81], winners_by_match[82]),
            95: (winners_by_match[86], winners_by_match[88]),
            96: (winners_by_match[85], winners_by_match[87]),
        }

        for mid in range(89, 97):
            t1, t2 = r16[mid]
            winner = _knockout_match_idx(t1, t2, lam_home, lam_away, elo_arr, rng)
            winners_by_match[mid] = winner
            losers_by_match[mid] = t2 if winner == t1 else t1
            path_counters["R16"][mid][(t1, t2, winner)] += 1
            reach_qf[winner] += 1

        qf = {
            97: (winners_by_match[89], winners_by_match[90]),
            98: (winners_by_match[93], winners_by_match[94]),
            99: (winners_by_match[91], winners_by_match[92]),
            100: (winners_by_match[95], winners_by_match[96]),
        }

        for mid in range(97, 101):
            t1, t2 = qf[mid]
            winner = _knockout_match_idx(t1, t2, lam_home, lam_away, elo_arr, rng)
            winners_by_match[mid] = winner
            losers_by_match[mid] = t2 if winner == t1 else t1
            path_counters["QF"][mid][(t1, t2, winner)] += 1
            reach_sf[winner] += 1

        sf = {
            101: (winners_by_match[97], winners_by_match[98]),
            102: (winners_by_match[99], winners_by_match[100]),
        }

        for mid in range(101, 103):
            t1, t2 = sf[mid]
            winner = _knockout_match_idx(t1, t2, lam_home, lam_away, elo_arr, rng)
            winners_by_match[mid] = winner
            losers_by_match[mid] = t2 if winner == t1 else t1
            path_counters["SF"][mid][(t1, t2, winner)] += 1

        finalists = [winners_by_match[101], winners_by_match[102]]
        for team_idx in finalists:
            reach_final[team_idx] += 1

        third_winner = _knockout_match_idx(losers_by_match[101], losers_by_match[102], lam_home, lam_away, elo_arr, rng)
        path_counters["3P"][103][(losers_by_match[101], losers_by_match[102], third_winner)] += 1

        champion = _knockout_match_idx(winners_by_match[101], winners_by_match[102], lam_home, lam_away, elo_arr, rng)
        path_counters["Final"][104][(winners_by_match[101], winners_by_match[102], champion)] += 1
        win_tournament[champion] += 1

        latest_stats = stats

    probs = pd.DataFrame(
        {
            "Team": ALL_TEAMS,
            "Group": [TEAM_GROUP[t] for t in ALL_TEAMS],
            "Advance to R32 %": np.round(100.0 * advance_r32 / iterations, 2),
            "Reach Round of 16 %": np.round(100.0 * reach_r16 / iterations, 2),
            "Reach Quarter-finals %": np.round(100.0 * reach_qf / iterations, 2),
            "Reach Semi-finals %": np.round(100.0 * reach_sf / iterations, 2),
            "Reach Final %": np.round(100.0 * reach_final / iterations, 2),
            "Win Tournament %": np.round(100.0 * win_tournament / iterations, 2),
            "Top 2 in Group %": np.round(100.0 * finish_top2 / iterations, 2),
            "Best Third %": np.round(100.0 * finish_best3 / iterations, 2),
        }
    ).sort_values("Win Tournament %", ascending=False)

    most_likely_path = {}
    for stage, match_map in path_counters.items():
        most_likely_path[stage] = {}
        for mid, counter in match_map.items():
            (t1_idx, t2_idx, winner_idx), count = counter.most_common(1)[0]
            label = f"{ALL_TEAMS[t1_idx]} vs {ALL_TEAMS[t2_idx]} -> {ALL_TEAMS[winner_idx]}"
            most_likely_path[stage][mid] = {"label": label, "prob": round(100.0 * count / iterations, 2)}

    latest_group_table = _build_group_table_from_stats(latest_stats, elo_arr)
    modal_pts = np.argmax(pts_freq, axis=1)          # whole number, 0-9
    modal_gd  = np.argmax(gd_freq, axis=1) - 12      # whole number, -12 to +12
    modal_gf  = np.argmax(gf_freq, axis=1)           # whole number, 0-19
    predicted_group_table = _build_predicted_group_table(modal_pts, modal_gd, modal_gf, elo_arr)

    return {
        "probabilities": probs,
        "tournament_wins": {team: int(count) for team, count in zip(ALL_TEAMS, win_tournament)},
        "most_likely_path": most_likely_path,
        "stage_path_counters": path_counters,
        "latest_group_table": latest_group_table,
        "predicted_group_table": predicted_group_table,
    }


def _build_predicted_group_table(
    modal_pts: np.ndarray,
    modal_gd: np.ndarray,
    modal_gf: np.ndarray,
    elo_arr: np.ndarray,
) -> pd.DataFrame:
    """Build predicted final group standings using the modal (most-likely) value
    from the frequency distribution — always whole numbers and achievable."""
    rows = []
    for group in GROUP_ORDER:
        team_indices = GROUP_TEAM_INDICES[group]
        pts = modal_pts[team_indices]
        gd  = modal_gd[team_indices]
        gf  = modal_gf[team_indices]
        elo = elo_arr[team_indices]
        order = np.lexsort((-elo, -gf, -gd, -pts))
        for pos, local_i in enumerate(order, 1):
            ti = int(team_indices[local_i])
            rows.append({
                "Group": group,
                "Pos": pos,
                "Team": ALL_TEAMS[ti],
                "Pred Pts": int(modal_pts[ti]),
                "Pred GD":  int(modal_gd[ti]),
                "Pred GF":  int(modal_gf[ti]),
            })
    return pd.DataFrame(rows)


def build_team_stage_opponents(
    stage_path_counters: Dict[str, Dict[int, Counter]],
    selected_team: str,
) -> pd.DataFrame:
    team_idx = TEAM_INDEX[selected_team]
    stage_map = [
        ("Round of 32", "R32"),
        ("Round of 16", "R16"),
        ("Quarter-finals", "QF"),
        ("Semi-finals", "SF"),
        ("Final", "Final"),
    ]

    rows = []
    for stage_label, stage_key in stage_map:
        opponent_counts: Counter = Counter()
        stage_total = 0

        for counter in stage_path_counters.get(stage_key, {}).values():
            for (t1_idx, t2_idx, _winner_idx), count in counter.items():
                if t1_idx == team_idx:
                    opponent_counts[t2_idx] += count
                    stage_total += count
                elif t2_idx == team_idx:
                    opponent_counts[t1_idx] += count
                    stage_total += count

        top_three = opponent_counts.most_common(3)
        if not top_three:
            rows.append({"Stage": stage_label, "Most likely opponents (top 3)": "No appearances in simulations"})
            continue

        formatted = []
        for opp_idx, count in top_three:
            pct = 100.0 * count / stage_total if stage_total else 0.0
            formatted.append(f"{ALL_TEAMS[opp_idx]} ({pct:.1f}%)")

        rows.append({"Stage": stage_label, "Most likely opponents (top 3)": ", ".join(formatted)})

    return pd.DataFrame(rows)


def render_group_tables(table_df: pd.DataFrame) -> None:
    cols = ["Group", "Pos", "Team", "Pts", "GD", "GF", "GA", "W", "D", "L", "P"]
    for group in GROUPS:
        st.subheader(f"Group {group}")
        st.dataframe(table_df[table_df["Group"] == group][cols], use_container_width=True, hide_index=True)


def render_dual_group_tables(live_df: pd.DataFrame, predicted_df: pd.DataFrame) -> None:
    """Show live standings and MC-predicted final standings side-by-side per group."""
    live_cols = ["Pos", "Team", "Pts", "GD", "GF", "GA", "W", "D", "L", "P"]
    pred_cols = ["Pos", "Team", "Pred Pts", "Pred GD", "Pred GF"]
    for group in GROUPS:
        st.markdown(f"**Group {group}**")
        col_live, col_pred = st.columns(2)
        with col_live:
            st.caption("🔴 Live / completed results")
            gdf = live_df[live_df["Group"] == group][live_cols]
            st.dataframe(gdf, use_container_width=True, hide_index=True)
        with col_pred:
            st.caption("🔮 Predicted final (MC average)")
            pdf = predicted_df[predicted_df["Group"] == group][pred_cols]
            st.dataframe(pdf, use_container_width=True, hide_index=True)


def render_predicted_thirds(predicted_df: pd.DataFrame, probs_df: pd.DataFrame) -> None:
    """Ranked table of all 12 predicted third-place teams.

    Ranking follows the actual FIFA rule: Pts → GD → GF for the 8 best third-place
    spots. 'Best Third %' is shown as context — it reflects how often a team actually
    ends up in third place across simulations (teams that usually finish top-2 will
    show a low % here even though their stats are good when they do finish third).
    """
    thirds = predicted_df[predicted_df["Pos"] == 3][
        ["Group", "Team", "Pred Pts", "Pred GD", "Pred GF"]
    ].copy()

    # Sort by FIFA third-place ranking criteria: Pts DESC, GD DESC, GF DESC.
    thirds = thirds.sort_values(
        ["Pred Pts", "Pred GD", "Pred GF"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    thirds.insert(0, "Rank", range(1, len(thirds) + 1))
    # Top 8 by stats advance — same rule FIFA applies on matchday 3.
    thirds["Advances?"] = thirds["Rank"].apply(lambda r: "✅ Yes" if r <= 8 else "❌ No")

    # Join probability context columns (informational only).
    probs_sub = probs_df[["Team", "Best Third %", "Top 2 in Group %"]].set_index("Team")
    thirds = thirds.set_index("Team").join(probs_sub).reset_index()
    thirds = thirds.rename(columns={"Top 2 in Group %": "Finishes Top 2 %"})

    st.caption(
        "ℹ️ **Ranked by predicted Pts → GD → GF** (the actual FIFA tiebreaker rule). "
        "Teams ranked 1–8 are predicted to advance as best third-place finishers. "
        "'Best Third %' shows how often each team finishes 3rd in their group across simulations — "
        "teams like Japan have strong stats when they do finish 3rd, but they usually "
        "finish 1st or 2nd instead (see 'Finishes Top 2 %')."
    )
    st.dataframe(
        thirds[["Rank", "Group", "Team", "Pred Pts", "Pred GD", "Pred GF",
                "Finishes Top 2 %", "Best Third %", "Advances?"]],
        use_container_width=True, hide_index=True,
    )


def build_projected_r32_fixtures(predicted_df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    winners = predicted_df[predicted_df["Pos"] == 1].set_index("Group")["Team"].to_dict()
    runners = predicted_df[predicted_df["Pos"] == 2].set_index("Group")["Team"].to_dict()

    thirds = predicted_df[predicted_df["Pos"] == 3][
        ["Group", "Team", "Pred Pts", "Pred GD", "Pred GF"]
    ].copy()
    thirds = thirds.sort_values(
        ["Pred Pts", "Pred GD", "Pred GF", "Group"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    best_thirds = thirds.head(8)

    third_slots = assign_third_place_slots(best_thirds)
    combo_key = "".join(sorted(best_thirds["Group"].tolist()))

    fixtures = [
        ("M73", runners["A"], runners["B"]),
        ("M74", winners["E"], third_slots["M74"]),
        ("M75", winners["F"], runners["C"]),
        ("M76", winners["C"], runners["F"]),
        ("M77", winners["I"], third_slots["M77"]),
        ("M78", runners["E"], runners["I"]),
        ("M79", winners["A"], third_slots["M79"]),
        ("M80", winners["L"], third_slots["M80"]),
        ("M81", winners["D"], third_slots["M81"]),
        ("M82", winners["G"], third_slots["M82"]),
        ("M83", runners["K"], runners["L"]),
        ("M84", winners["H"], runners["J"]),
        ("M85", winners["B"], third_slots["M85"]),
        ("M86", winners["J"], runners["H"]),
        ("M87", winners["K"], third_slots["M87"]),
        ("M88", runners["D"], runners["G"]),
    ]

    fixture_rows = [{"Match": m, "Home": home, "Away": away} for m, home, away in fixtures]
    return pd.DataFrame(fixture_rows), combo_key


def _estimate_knockout_win_probability(
    team_a: str,
    team_b: str,
    strength: pd.DataFrame,
    rng: np.random.Generator,
    trials: int = 1500,
) -> Tuple[str, float]:
    lam_a, lam_b = expected_goals(team_a, team_b, strength)
    wins_a = 0
    wins_b = 0

    for _ in range(trials):
        g_a = int(rng.poisson(lam_a))
        g_b = int(rng.poisson(lam_b))

        if g_a == g_b:
            g_a += int(rng.poisson(lam_a * 0.33))
            g_b += int(rng.poisson(lam_b * 0.33))

            if g_a == g_b:
                elo_diff = strength.loc[team_a, "elo"] - strength.loc[team_b, "elo"]
                p_team_a = 1.0 / (1.0 + math.exp(-elo_diff / 115.0))
                if rng.random() < p_team_a:
                    wins_a += 1
                else:
                    wins_b += 1
                continue

        if g_a > g_b:
            wins_a += 1
        else:
            wins_b += 1

    winner = team_a if wins_a >= wins_b else team_b
    winner_prob = 100.0 * max(wins_a, wins_b) / max(1, wins_a + wins_b)
    return winner, round(winner_prob, 2)


def build_projected_bracket_path(
    projected_r32: pd.DataFrame,
    strength: pd.DataFrame,
    seed: int = 20260620,
) -> pd.DataFrame:
    fixtures = projected_r32.set_index("Match")
    winners_by_match: Dict[int, str] = {}
    cumulative = 1.0
    rows = []
    rng = np.random.default_rng(seed)

    def add_row(stage: str, mid: int, home: str, away: str) -> None:
        nonlocal cumulative
        winner = knockout_match(home, away, strength, rng)[0]
        _, win_prob = _estimate_knockout_win_probability(home, away, strength, rng, trials=600)
        cumulative *= win_prob / 100.0
        winners_by_match[mid] = winner
        rows.append(
            {
                "Stage": stage,
                "Match": f"M{mid}",
                "Home": home,
                "Away": away,
                "Projected Winner": winner,
                "Win Probability %": win_prob,
                "Cumulative Path Probability %": round(cumulative * 100.0, 6),
            }
        )

    for mid in range(73, 89):
        match = f"M{mid}"
        add_row("Round of 32", mid, fixtures.loc[match, "Home"], fixtures.loc[match, "Away"])

    r16_pairs = {
        89: (winners_by_match[74], winners_by_match[77]),
        90: (winners_by_match[73], winners_by_match[75]),
        91: (winners_by_match[76], winners_by_match[78]),
        92: (winners_by_match[79], winners_by_match[80]),
        93: (winners_by_match[83], winners_by_match[84]),
        94: (winners_by_match[81], winners_by_match[82]),
        95: (winners_by_match[86], winners_by_match[88]),
        96: (winners_by_match[85], winners_by_match[87]),
    }
    for mid in range(89, 97):
        home, away = r16_pairs[mid]
        add_row("Round of 16", mid, home, away)

    qf_pairs = {
        97: (winners_by_match[89], winners_by_match[90]),
        98: (winners_by_match[93], winners_by_match[94]),
        99: (winners_by_match[91], winners_by_match[92]),
        100: (winners_by_match[95], winners_by_match[96]),
    }
    for mid in range(97, 101):
        home, away = qf_pairs[mid]
        add_row("Quarter-finals", mid, home, away)

    sf_pairs = {
        101: (winners_by_match[97], winners_by_match[98]),
        102: (winners_by_match[99], winners_by_match[100]),
    }
    for mid in range(101, 103):
        home, away = sf_pairs[mid]
        add_row("Semi-finals", mid, home, away)

    final_home = winners_by_match[101]
    final_away = winners_by_match[102]
    add_row("Final", 104, final_home, final_away)

    return pd.DataFrame(rows)


def render_bracket_flow_diagram(projected_path: pd.DataFrame) -> None:
    rows = projected_path.set_index("Match").to_dict(orient="index")

    def label_for(match: str) -> str:
        row = rows[match]
        return (
            f"{match}<br/>"
            f"{row['Home']} vs {row['Away']}<br/>"
            f"Winner: {row['Projected Winner']}<br/>"
            f"{row['Win Probability %']:.1f}%"
        )

    stages = [
        ("R32", [f"M{i}" for i in range(73, 89)]),
        ("R16", [f"M{i}" for i in range(89, 97)]),
        ("QF", [f"M{i}" for i in range(97, 101)]),
        ("SF", [f"M{i}" for i in range(101, 103)]),
        ("Final", ["M104"]),
    ]

    next_map = {
        "M73": ["M90"],
        "M75": ["M90"],
        "M76": ["M91"],
        "M78": ["M91"],
        "M79": ["M92"],
        "M80": ["M92"],
        "M83": ["M93"],
        "M84": ["M93"],
        "M81": ["M94"],
        "M82": ["M94"],
        "M86": ["M95"],
        "M88": ["M95"],
        "M85": ["M96"],
        "M87": ["M96"],
        "M74": ["M89"],
        "M77": ["M89"],
        "M89": ["M97"],
        "M90": ["M97"],
        "M93": ["M98"],
        "M94": ["M98"],
        "M91": ["M99"],
        "M92": ["M99"],
        "M95": ["M100"],
        "M96": ["M100"],
        "M97": ["M101"],
        "M98": ["M101"],
        "M99": ["M102"],
        "M100": ["M102"],
        "M101": ["M104"],
        "M102": ["M104"],
    }

    lines = [
        "<div style='border:1px solid #e6e8eb;border-radius:14px;padding:14px;background:linear-gradient(180deg,#fff 0%,#f8fafc 100%);'>",
        "<pre class='mermaid'>",
        "flowchart TD",
    ]

    for stage_name, match_ids in stages:
        lines.append(f'  subgraph {stage_name}')
        lines.append("  direction TB")
        for match in match_ids:
            lines.append(f'    {match.lower()}["{label_for(match)}"]')
        lines.append("  end")

    for src, targets in next_map.items():
        for dst in targets:
            lines.append(f"  {src.lower()} --> {dst.lower()}")

    lines.extend([
        "  style m104 fill:#0f4e94,color:#ffffff,stroke:#0f4e94,stroke-width:2px",
        "</pre>",
        "<script type='module'>",
        "import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';",
        "mermaid.initialize({ startOnLoad: true, theme: 'neutral', securityLevel: 'loose' });",
        "</script>",
        "</div>",
    ])

    st.html("\n".join(lines))


def render_most_likely_path(path: Dict) -> None:
    stage_order = [
        ("R32", "Round of 32", range(73, 89)),
        ("R16", "Round of 16", range(89, 97)),
        ("QF", "Quarter-finals", range(97, 101)),
        ("SF", "Semi-finals", range(101, 103)),
        ("3P", "Third-place play-off", [103]),
        ("Final", "Final", [104]),
    ]

    for key, title, mids in stage_order:
        st.subheader(title)
        rows = []
        for mid in mids:
            item = path[key][mid]
            rows.append({"Match": f"M{mid}", "Most likely result": item["label"], "Path probability %": item["prob"]})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def main() -> None:
    st.title("FIFA World Cup 2026 Live Dashboard")
    st.caption("Live ESPN ingestion + Elo/xG Poisson model + scalable Monte Carlo projection")

    st.sidebar.header("Controls")
    iterations = st.sidebar.number_input("Monte Carlo iterations", min_value=1000, max_value=50000, value=1000, step=1000)
    selected_team = st.sidebar.selectbox("Team spotlight", ALL_TEAMS, index=ALL_TEAMS.index("Scotland"))
    run_button = st.sidebar.button("Fetch Live Scores & Run Simulation", type="primary", use_container_width=True)

    if run_button:
        with st.spinner("Fetching ESPN live scores and running Monte Carlo..."):
            payload = fetch_espn_scoreboard()
            espn_completed, live_df, warnings = parse_espn_matches(payload)
            locked_lookup = merge_locked_results(espn_completed)
            locked_rows = [(g, h, a, gh, ga) for (g, h, a), (gh, ga) in locked_lookup.items()]
            sim = run_monte_carlo(locked_rows, int(iterations), 20260620)
            live_table = locked_group_tables(locked_lookup, build_strength_table())

        st.session_state["dashboard_result"] = {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "live_df": live_df,
            "warnings": warnings,
            "espn_completed_count": len(espn_completed),
            "locked_group_table": live_table,
            "sim": sim,
        }

    result = st.session_state.get("dashboard_result")
    if result is None:
        st.info("Click 'Fetch Live Scores & Run Simulation' in the sidebar to load live data and run the projection (default 1,000 iterations; increase to 10,000 when ready).")
        return

    st.sidebar.markdown(f"Last update: {result['timestamp']}")

    team_probs = result["sim"]["probabilities"].set_index("Team")
    if selected_team in team_probs.index:
        row = team_probs.loc[selected_team]
        st.sidebar.subheader(f"{selected_team} spotlight")
        st.sidebar.metric("Advance to Round of 32", f"{row['Advance to R32 %']:.2f}%")
        st.sidebar.metric("Top 2 in group", f"{row['Top 2 in Group %']:.2f}%")
        st.sidebar.metric("Reach Round of 16", f"{row['Reach Round of 16 %']:.2f}%")
        st.sidebar.metric("Reach Quarter-finals", f"{row['Reach Quarter-finals %']:.2f}%")
        st.sidebar.metric("Reach Semi-finals", f"{row['Reach Semi-finals %']:.2f}%")
        st.sidebar.metric("Reach Final", f"{row['Reach Final %']:.2f}%")
        st.sidebar.metric("Win Tournament", f"{row['Win Tournament %']:.2f}%")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Live Standings",
        "Tournament Probabilities",
        "Knockout Bracket Most Likely Path",
        "🏆 Team Path Explorer",
    ])

    with tab1:
        st.subheader("Group Standings")
        st.write(f"ESPN completed matches merged: **{result['espn_completed_count']}**")
        render_dual_group_tables(
            result["locked_group_table"],
            result["sim"]["predicted_group_table"],
        )

        st.markdown("---")
        st.subheader("Predicted third-place rankings")
        st.caption(
            "Ranked by Monte Carlo qualification frequency across all simulated tournaments. "
            "Teams ranked 1–8 are projected to advance as one of the 8 best third-place finishers."
        )
        render_predicted_thirds(
            result["sim"]["predicted_group_table"],
            result["sim"]["probabilities"],
        )

        st.markdown("---")
        st.subheader("Projected Round of 32 fixtures")
        projected_r32, combo_key = build_projected_r32_fixtures(result["sim"]["predicted_group_table"])
        st.caption(
            "Built from predicted group winners/runners-up plus top-8 predicted third-place teams, "
            f"then mapped through Appendix C combination **{combo_key}**."
        )
        st.caption(
            "Note: Round of 32 matchups can still pair teams from different groups that feel surprising at first glance. "
            "For example, a Group C third-place Scotland can be assigned to Appendix C slot M80, which is the Group L winner slot. "
            "That is why England can legitimately appear as a possible R32 opponent in this bracket model."
        )
        st.dataframe(projected_r32, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("Projected path to the Final")
        projected_path = build_projected_bracket_path(
            projected_r32,
            build_strength_table(),
        )
        st.caption(
            "Each round is simulated from the projected bracket using the matchup model; "
            "win probability is the estimated chance that the projected winner beats that opponent."
        )
        render_bracket_flow_diagram(projected_path)
        st.dataframe(projected_path, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("Currently live matches")
        if result["live_df"].empty:
            st.write("No in-progress matches reported at the latest fetch.")
        else:
            st.dataframe(result["live_df"], use_container_width=True, hide_index=True)

        if result["warnings"]:
            with st.expander("Parser warnings"):
                for warning in result["warnings"]:
                    st.write(f"- {warning}")

    with tab2:
        probs = result["sim"]["probabilities"].copy()
        display_cols = [
            "Team",
            "Group",
            "Advance to R32 %",
            "Reach Round of 16 %",
            "Reach Quarter-finals %",
            "Reach Semi-finals %",
            "Reach Final %",
            "Win Tournament %",
        ]

        st.dataframe(probs[display_cols], use_container_width=True, hide_index=True)

        st.subheader(f"{selected_team}: exact probabilistic path")
        if selected_team in team_probs.index:
            row = team_probs.loc[selected_team]
            spotlight = pd.DataFrame(
                [
                    {"Stage": "Top 2 in Group", "Probability %": row["Top 2 in Group %"]},
                    {"Stage": "Best Third-place Qualifier", "Probability %": row["Best Third %"]},
                    {"Stage": "Advance to Round of 32", "Probability %": row["Advance to R32 %"]},
                    {"Stage": "Reach Round of 16", "Probability %": row["Reach Round of 16 %"]},
                    {"Stage": "Reach Quarter-finals", "Probability %": row["Reach Quarter-finals %"]},
                    {"Stage": "Reach Semi-finals", "Probability %": row["Reach Semi-finals %"]},
                    {"Stage": "Reach Final", "Probability %": row["Reach Final %"]},
                    {"Stage": "Win Tournament", "Probability %": row["Win Tournament %"]},
                ]
            )
            st.dataframe(spotlight, use_container_width=True, hide_index=True)

    with tab3:
        render_most_likely_path(result["sim"]["most_likely_path"])

    with tab4:
        st.subheader(f"{selected_team} — knockout path explorer")

        if selected_team in team_probs.index:
            row = team_probs.loc[selected_team]
            team_group = TEAM_GROUP[selected_team]

            # ── Group snapshot ─────────────────────────────────────────────
            st.markdown("**Current group snapshot**")
            group_snap = result["locked_group_table"][
                result["locked_group_table"]["Group"] == team_group
            ][["Pos", "Team", "Pts", "GD", "GF", "GA", "W", "D", "L", "P"]]
            st.dataframe(group_snap, use_container_width=True, hide_index=True)

            # ── Group rivals advance-probability bar chart ──────────────────
            st.markdown("**Projected Round of 32 advance probability — all Group " + team_group + " teams**")
            group_rivals_prob = (
                result["sim"]["probabilities"]
                .loc[result["sim"]["probabilities"]["Group"] == team_group,
                     ["Team", "Advance to R32 %"]]
                .sort_values("Advance to R32 %", ascending=False)
                .set_index("Team")
            )
            st.bar_chart(group_rivals_prob)

            # ── Biggest group threat callout ────────────────────────────────
            rivals = group_rivals_prob.drop(index=selected_team, errors="ignore")
            if not rivals.empty:
                top_rival = str(rivals.index[0])
                top_rival_pct = float(rivals.iloc[0, 0])
                own_pct = float(row["Advance to R32 %"])
                delta = top_rival_pct - own_pct
                if delta > 0:
                    threat_color = "#d9534f"
                    threat_msg = (
                        f"⚠️ Biggest group threat: <strong>{top_rival}</strong> &mdash; "
                        f"R32 advance probability <strong>{top_rival_pct:.1f}%</strong> "
                        f"(<strong>+{delta:.1f}%</strong> vs {selected_team})"
                    )
                else:
                    threat_color = "#5cb85c"
                    threat_msg = (
                        f"✅ <strong>{selected_team}</strong> leads all group rivals in projected R32 advance — "
                        f"nearest rival: <strong>{top_rival}</strong> ({top_rival_pct:.1f}%)"
                    )
                st.markdown(
                    f"<div style='border-left:4px solid {threat_color};padding:8px 14px;"
                    f"background:#fafafa;border-radius:4px;margin:10px 0;'>{threat_msg}</div>",
                    unsafe_allow_html=True,
                )

            # ── Knockout stage probability tree with progress bars ──────────
            st.markdown("---")
            st.markdown("**Knockout stage probability tree**")

            tier_blocks = [
                ("Round of 32",      float(row["Advance to R32 %"]),       "#4a90d9"),
                ("Round of 16",      float(row["Reach Round of 16 %"]),     "#2e78c9"),
                ("Quarter-finals",   float(row["Reach Quarter-finals %"]),  "#1a61b0"),
                ("Semi-finals",      float(row["Reach Semi-finals %"]),     "#0f4e94"),
                ("Final",            float(row["Reach Final %"]),            "#08347a"),
                ("Win Tournament 🏆", float(row["Win Tournament %"]),       "#f5a623"),
            ]

            for tier_name, tier_prob, color in tier_blocks:
                col_text, col_bar = st.columns([2, 3])
                with col_text:
                    st.markdown(
                        f"<div style='padding:6px 0;'>"
                        f"<span style='font-weight:600;font-size:15px;'>{tier_name}</span><br/>"
                        f"<span style='font-size:22px;font-weight:700;color:{color};'>"
                        f"{tier_prob:.1f}%</span></div>",
                        unsafe_allow_html=True,
                    )
                with col_bar:
                    st.progress(max(0.0, min(1.0, tier_prob / 100.0)))

        st.markdown("---")
        st.subheader("Most likely opponents by stage")
        st.caption(
            "These are stage-specific opponents from the simulation tree. The Round of 32 row only counts actual R32 pairings; "
            "later rounds are reported separately, so this table should not be read as a generic 'any time in the tournament' list."
        )
        opponents_df = build_team_stage_opponents(result["sim"]["stage_path_counters"], selected_team)
        st.dataframe(opponents_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
