import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.stats import poisson


# Data snapshot assumptions:
# - Date: 2026-06-20 (before the 20 June matches kick off)
# - Matches listed in COMPLETED_MATCHES were finished by end of 19 June.

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

# Full round-robin fixture list per group.
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

# Hardcoded completed matches through 19 June 2026.
COMPLETED_MATCHES: List[Tuple[str, str, str, int, int]] = [
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

# Approximate FIFA rank snapshot proxy.
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

# Historical xG priors (approximate), for teams where stronger priors are useful.
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


@dataclass
class MatchResult:
    group: Optional[str]
    home: str
    away: str
    home_goals: int
    away_goals: int
    stage: str
    note: str = ""


def build_team_strengths() -> pd.DataFrame:
    teams = sorted({team for teams in GROUPS.values() for team in teams})

    rows = []
    for team in teams:
        rank = FIFA_RANK[team]
        elo = 1975 - 7.0 * rank

        if team in XG_OVERRIDES:
            xg_for, xg_against = XG_OVERRIDES[team]
        else:
            # Derived priors for teams without specific override.
            xg_for = np.clip(1.00 + (elo - 1550) / 430, 0.80, 1.80)
            xg_against = np.clip(1.62 - (elo - 1550) / 520, 0.90, 1.80)

        rows.append(
            {
                "team": team,
                "rank": rank,
                "elo": elo,
                "xg_for": float(xg_for),
                "xg_against": float(xg_against),
            }
        )

    df = pd.DataFrame(rows).set_index("team")
    return df


def expected_goals(home: str, away: str, strength: pd.DataFrame) -> Tuple[float, float]:
    s_h = strength.loc[home]
    s_a = strength.loc[away]

    # Composite strength model using Elo, FIFA ranking and xG priors.
    elo_term_h = (s_h["elo"] - s_a["elo"]) / 400.0
    elo_term_a = -elo_term_h

    rank_term_h = (s_a["rank"] - s_h["rank"]) / 60.0
    rank_term_a = -rank_term_h

    xg_attack_h = s_h["xg_for"] / strength["xg_for"].mean()
    xg_def_opp_h = s_a["xg_against"] / strength["xg_against"].mean()

    xg_attack_a = s_a["xg_for"] / strength["xg_for"].mean()
    xg_def_opp_a = s_h["xg_against"] / strength["xg_against"].mean()

    base = 1.33

    lam_h = base * math.exp(0.47 * elo_term_h + 0.22 * rank_term_h) * xg_attack_h * xg_def_opp_h
    lam_a = base * math.exp(0.47 * elo_term_a + 0.22 * rank_term_a) * xg_attack_a * xg_def_opp_a

    lam_h = float(np.clip(lam_h, 0.15, 3.80))
    lam_a = float(np.clip(lam_a, 0.15, 3.80))

    return lam_h, lam_a


def simulate_score(home: str, away: str, strength: pd.DataFrame, rng: np.random.Generator) -> Tuple[int, int]:
    lam_h, lam_a = expected_goals(home, away, strength)
    gh = int(poisson.rvs(lam_h, random_state=rng))
    ga = int(poisson.rvs(lam_a, random_state=rng))
    return gh, ga


def init_table(group: str) -> Dict[str, Dict[str, int]]:
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


def sort_group_table(
    group: str,
    table: Dict[str, Dict[str, int]],
    strength: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for team, stats in table.items():
        row = {"Group": group, "Team": team}
        row.update(stats)
        row["Elo"] = float(strength.loc[team, "elo"])
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values(
        by=["Pts", "GD", "GF", "Elo"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    df["Pos"] = df.index + 1
    return df[["Group", "Pos", "Team", "Pts", "GD", "GF", "GA", "W", "D", "L", "P"]]


def simulate_group_stage(strength: pd.DataFrame, rng: np.random.Generator):
    completed_lookup = {
        (g, h, a): (gh, ga)
        for g, h, a, gh, ga in COMPLETED_MATCHES
    }

    all_tables = []
    all_results: List[MatchResult] = []

    for g in GROUPS.keys():
        table = init_table(g)

        for home, away in GROUP_FIXTURES[g]:
            key = (g, home, away)
            if key in completed_lookup:
                gh, ga = completed_lookup[key]
                note = "played"
            else:
                gh, ga = simulate_score(home, away, strength, rng)
                note = "simulated"

            apply_result(table, home, away, gh, ga)
            all_results.append(
                MatchResult(
                    group=g,
                    home=home,
                    away=away,
                    home_goals=gh,
                    away_goals=ga,
                    stage="Group",
                    note=note,
                )
            )

        gdf = sort_group_table(g, table, strength)
        all_tables.append(gdf)

    group_table = pd.concat(all_tables, ignore_index=True)

    winners = group_table[group_table["Pos"] == 1].set_index("Group")["Team"].to_dict()
    runners = group_table[group_table["Pos"] == 2].set_index("Group")["Team"].to_dict()
    thirds_df = group_table[group_table["Pos"] == 3].copy()
    thirds_df = thirds_df.sort_values(by=["Pts", "GD", "GF"], ascending=[False, False, False]).reset_index(drop=True)
    best_thirds = thirds_df.head(8).copy()

    return group_table, winners, runners, best_thirds, all_results


def assign_third_place_slots(best_thirds: pd.DataFrame) -> Dict[str, str]:
    # Slot constraints taken from FIFA fixture placeholders.
    slot_allowed_groups = {
        "M74": set("ABCDF"),
        "M77": set("CDFGH"),
        "M79": set("CEFHI"),
        "M80": set("EHIJK"),
        "M81": set("BEFIJ"),
        "M82": set("AEHIJ"),
        "M85": set("EFGIJ"),
        "M87": set("DEIJL"),
    }

    group_to_team = {
        row["Group"]: row["Team"]
        for _, row in best_thirds.iterrows()
    }

    qualified_groups = set(group_to_team.keys())
    slots = list(slot_allowed_groups.keys())

    # Candidate groups for each slot, restricted to qualified third-place groups.
    candidates = {
        slot: sorted(list(slot_allowed_groups[slot] & qualified_groups))
        for slot in slots
    }

    for slot, cands in candidates.items():
        if not cands:
            raise RuntimeError(f"No valid third-place group candidates for slot {slot}")

    slot_order = sorted(slots, key=lambda s: len(candidates[s]))

    def backtrack(i: int, used: set, assign: Dict[str, str]) -> Optional[Dict[str, str]]:
        if i == len(slot_order):
            return assign.copy()

        slot = slot_order[i]
        for grp in candidates[slot]:
            if grp in used:
                continue
            assign[slot] = grp
            used.add(grp)
            res = backtrack(i + 1, used, assign)
            if res is not None:
                return res
            used.remove(grp)
            del assign[slot]

        return None

    group_assignment = backtrack(0, set(), {})
    if group_assignment is None:
        raise RuntimeError("Unable to assign third-place teams to Round-of-32 slots")

    slot_to_team = {slot: group_to_team[grp] for slot, grp in group_assignment.items()}
    return slot_to_team


def knockout_match(
    team_a: str,
    team_b: str,
    strength: pd.DataFrame,
    rng: np.random.Generator,
    stage: str,
) -> Tuple[str, MatchResult]:
    lam_a, lam_b = expected_goals(team_a, team_b, strength)

    g_a = int(poisson.rvs(lam_a, random_state=rng))
    g_b = int(poisson.rvs(lam_b, random_state=rng))

    if g_a > g_b:
        winner = team_a
        note = "90m"
    elif g_b > g_a:
        winner = team_b
        note = "90m"
    else:
        # Extra time model: 30 minutes -> approx one-third of regular intensity.
        et_a = int(poisson.rvs(lam_a * 0.33, random_state=rng))
        et_b = int(poisson.rvs(lam_b * 0.33, random_state=rng))

        g_a += et_a
        g_b += et_b

        if et_a != et_b:
            winner = team_a if g_a > g_b else team_b
            note = "AET"
        else:
            # Penalties weighted by relative team strength (mostly Elo).
            elo_diff = strength.loc[team_a, "elo"] - strength.loc[team_b, "elo"]
            p_team_a = 1.0 / (1.0 + math.exp(-elo_diff / 115.0))
            winner = team_a if rng.random() < p_team_a else team_b
            note = "PEN"

    result = MatchResult(
        group=None,
        home=team_a,
        away=team_b,
        home_goals=g_a,
        away_goals=g_b,
        stage=stage,
        note=note,
    )

    return winner, result


def simulate_knockout(
    winners: Dict[str, str],
    runners: Dict[str, str],
    best_thirds: pd.DataFrame,
    strength: pd.DataFrame,
    rng: np.random.Generator,
):
    third_slots = assign_third_place_slots(best_thirds)

    m = {}
    m[73] = (runners["A"], runners["B"])
    m[74] = (winners["E"], third_slots["M74"])
    m[75] = (winners["F"], runners["C"])
    m[76] = (winners["C"], runners["F"])
    m[77] = (winners["I"], third_slots["M77"])
    m[78] = (runners["E"], runners["I"])
    m[79] = (winners["A"], third_slots["M79"])
    m[80] = (winners["L"], third_slots["M80"])
    m[81] = (winners["D"], third_slots["M81"])
    m[82] = (winners["G"], third_slots["M82"])
    m[83] = (runners["K"], runners["L"])
    m[84] = (winners["H"], runners["J"])
    m[85] = (winners["B"], third_slots["M85"])
    m[86] = (winners["J"], runners["H"])
    m[87] = (winners["K"], third_slots["M87"])
    m[88] = (runners["D"], runners["G"])

    winners_by_match: Dict[int, str] = {}
    losers_by_match: Dict[int, str] = {}
    results_by_match: Dict[int, MatchResult] = {}

    for mid in range(73, 89):
        t1, t2 = m[mid]
        w, res = knockout_match(t1, t2, strength, rng, "R32")
        winners_by_match[mid] = w
        losers_by_match[mid] = t2 if w == t1 else t1
        results_by_match[mid] = res

    m89 = (winners_by_match[74], winners_by_match[77])
    m90 = (winners_by_match[73], winners_by_match[75])
    m91 = (winners_by_match[76], winners_by_match[78])
    m92 = (winners_by_match[79], winners_by_match[80])
    m93 = (winners_by_match[83], winners_by_match[84])
    m94 = (winners_by_match[81], winners_by_match[82])
    m95 = (winners_by_match[86], winners_by_match[88])
    m96 = (winners_by_match[85], winners_by_match[87])

    for mid, (t1, t2) in {
        89: m89,
        90: m90,
        91: m91,
        92: m92,
        93: m93,
        94: m94,
        95: m95,
        96: m96,
    }.items():
        w, res = knockout_match(t1, t2, strength, rng, "R16")
        winners_by_match[mid] = w
        losers_by_match[mid] = t2 if w == t1 else t1
        results_by_match[mid] = res

    for mid, (t1, t2) in {
        97: (winners_by_match[89], winners_by_match[90]),
        98: (winners_by_match[93], winners_by_match[94]),
        99: (winners_by_match[91], winners_by_match[92]),
        100: (winners_by_match[95], winners_by_match[96]),
    }.items():
        w, res = knockout_match(t1, t2, strength, rng, "QF")
        winners_by_match[mid] = w
        losers_by_match[mid] = t2 if w == t1 else t1
        results_by_match[mid] = res

    for mid, (t1, t2) in {
        101: (winners_by_match[97], winners_by_match[98]),
        102: (winners_by_match[99], winners_by_match[100]),
    }.items():
        w, res = knockout_match(t1, t2, strength, rng, "SF")
        winners_by_match[mid] = w
        losers_by_match[mid] = t2 if w == t1 else t1
        results_by_match[mid] = res

    # Third-place playoff
    w103, res103 = knockout_match(losers_by_match[101], losers_by_match[102], strength, rng, "3P")
    winners_by_match[103] = w103
    losers_by_match[103] = res103.away if w103 == res103.home else res103.home
    results_by_match[103] = res103

    # Final
    w104, res104 = knockout_match(winners_by_match[101], winners_by_match[102], strength, rng, "Final")
    winners_by_match[104] = w104
    losers_by_match[104] = res104.away if w104 == res104.home else res104.home
    results_by_match[104] = res104

    return third_slots, winners_by_match, results_by_match


def pretty_knockout_line(mid: int, result: MatchResult, winner: str) -> str:
    score = f"{result.home} {result.home_goals}-{result.away_goals} {result.away}"
    return f"Match {mid}: {score} | Winner: {winner} ({result.note})"


def main() -> None:
    rng = np.random.default_rng(20260620)
    strength = build_team_strengths()

    group_table, winners, runners, best_thirds, group_results = simulate_group_stage(strength, rng)

    third_slots, ko_winners, ko_results = simulate_knockout(
        winners=winners,
        runners=runners,
        best_thirds=best_thirds,
        strength=strength,
        rng=rng,
    )

    print("=" * 88)
    print("FIFA WORLD CUP 2026 FORECAST (as-of 2026-06-20)")
    print("Model: Elo + FIFA Rank Proxy + xG Priors + Poisson Match Simulation")
    print("=" * 88)

    print("\nFinal Projected Group Tables")
    print("-" * 88)
    out_cols = ["Group", "Pos", "Team", "Pts", "GD", "GF", "GA", "W", "D", "L", "P"]
    for g in GROUPS.keys():
        gdf = group_table[group_table["Group"] == g][out_cols]
        print(f"\nGroup {g}")
        print(gdf.to_string(index=False))

    auto_q = []
    for g in GROUPS.keys():
        gdf = group_table[group_table["Group"] == g].sort_values("Pos")
        auto_q.extend(gdf[gdf["Pos"].isin([1, 2])]["Team"].tolist())

    print("\n" + "-" * 88)
    print("Advancing Teams")
    print("-" * 88)
    print("Automatic qualifiers (top 2 from each group):")
    for i, t in enumerate(auto_q, start=1):
        print(f"{i:2d}. {t}")

    print("\nBest third-place qualifiers (8):")
    third_ranked = best_thirds[["Group", "Team", "Pts", "GD", "GF"]].copy()
    print(third_ranked.to_string(index=False))

    print("\nThird-place slot assignment into Round of 32 placeholders:")
    for slot in sorted(third_slots.keys()):
        print(f"{slot}: {third_slots[slot]}")

    print("\n" + "-" * 88)
    print("Knockout Bracket Prediction")
    print("-" * 88)

    print("\nRound of 32")
    for mid in range(73, 89):
        print(pretty_knockout_line(mid, ko_results[mid], ko_winners[mid]))

    print("\nRound of 16")
    for mid in range(89, 97):
        print(pretty_knockout_line(mid, ko_results[mid], ko_winners[mid]))

    print("\nQuarter-finals")
    for mid in range(97, 101):
        print(pretty_knockout_line(mid, ko_results[mid], ko_winners[mid]))

    print("\nSemi-finals")
    for mid in range(101, 103):
        print(pretty_knockout_line(mid, ko_results[mid], ko_winners[mid]))

    print("\nThird-place playoff")
    print(pretty_knockout_line(103, ko_results[103], ko_winners[103]))

    print("\nFinal")
    print(pretty_knockout_line(104, ko_results[104], ko_winners[104]))

    print("\n" + "-" * 88)
    print(f"Projected champion: {ko_winners[104]}")
    print(f"Projected runner-up: {ko_results[104].away if ko_winners[104] == ko_results[104].home else ko_results[104].home}")
    print(f"Projected third place: {ko_winners[103]}")
    print("-" * 88)


if __name__ == "__main__":
    main()
