import math
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
import app

ITERATIONS = 50000
SEED = 20260620
WORKERS = 5

ALL_TEAMS = app.ALL_TEAMS
TEAM_INDEX = app.TEAM_INDEX
GROUP_ORDER = app.GROUP_ORDER
GROUP_FIXTURE_INDICES = app.GROUP_FIXTURE_INDICES
GROUP_TEAM_INDICES = app.GROUP_TEAM_INDICES
STAT_PTS = app.STAT_PTS
STAT_GF = app.STAT_GF
STAT_GA = app.STAT_GA
STAT_W = app.STAT_W
STAT_D = app.STAT_D
STAT_L = app.STAT_L
STAT_P = app.STAT_P
SCOTLAND_IDX = TEAM_INDEX["Scotland"]

strength = app.build_strength_table()
LAM_HOME, LAM_AWAY, ELO_ARR = app._precompute_goal_rate_matrices(strength)
LOCKED_LOOKUP_IDX = {
    (g, TEAM_INDEX[h], TEAM_INDEX[a]): (gh, ga)
    for (g, h, a), (gh, ga) in app.merge_locked_results([]).items()
}


def knockout_match_idx(rng, team_a_idx, team_b_idx):
    g_a = int(rng.poisson(LAM_HOME[team_a_idx, team_b_idx]))
    g_b = int(rng.poisson(LAM_AWAY[team_a_idx, team_b_idx]))
    if g_a > g_b:
        return team_a_idx
    if g_b > g_a:
        return team_b_idx
    et_a = int(rng.poisson(LAM_HOME[team_a_idx, team_b_idx] * 0.33))
    et_b = int(rng.poisson(LAM_AWAY[team_a_idx, team_b_idx] * 0.33))
    g_a += et_a
    g_b += et_b
    if g_a != g_b:
        return team_a_idx if g_a > g_b else team_b_idx
    elo_diff = ELO_ARR[team_a_idx] - ELO_ARR[team_b_idx]
    p_team_a = 1.0 / (1.0 + math.exp(-elo_diff / 115.0))
    return team_a_idx if rng.random() < p_team_a else team_b_idx


def run_chunk(args):
    chunk_iterations, seed_offset = args
    rng = np.random.default_rng(SEED + seed_offset)
    scotland_wins = 0
    team_count = len(ALL_TEAMS)

    for _ in range(chunk_iterations):
        stats = np.zeros((team_count, 7), dtype=np.int16)
        winners_idx = {}
        runners_idx = {}
        third_rows = []

        for group in GROUP_ORDER:
            for home_idx, away_idx in GROUP_FIXTURE_INDICES[group]:
                key = (group, home_idx, away_idx)
                if key in LOCKED_LOOKUP_IDX:
                    gh, ga = LOCKED_LOOKUP_IDX[key]
                else:
                    gh = int(rng.poisson(LAM_HOME[home_idx, away_idx]))
                    ga = int(rng.poisson(LAM_AWAY[home_idx, away_idx]))

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

            team_indices = GROUP_TEAM_INDICES[group]
            pts = stats[team_indices, STAT_PTS]
            gd = stats[team_indices, STAT_GF] - stats[team_indices, STAT_GA]
            gf = stats[team_indices, STAT_GF]
            elo = ELO_ARR[team_indices]
            order = np.lexsort((-elo, -gf, -gd, -pts))
            sorted_group = team_indices[order]
            winners_idx[group] = int(sorted_group[0])
            runners_idx[group] = int(sorted_group[1])
            third_idx = int(sorted_group[2])
            third_rows.append((group, third_idx, int(stats[third_idx, STAT_PTS]), int(stats[third_idx, STAT_GF] - stats[third_idx, STAT_GA]), int(stats[third_idx, STAT_GF])))

        third_rows.sort(key=lambda x: (x[2], x[3], x[4]), reverse=True)
        best_third_groups = [row[0] for row in third_rows[:8]]
        best_third_indices = [row[1] for row in third_rows[:8]]
        third_slots_idx = app._assign_third_place_slots_idx(best_third_groups, best_third_indices)

        r32 = {
            73: (runners_idx["A"], runners_idx["B"]),
            74: (winners_idx["E"], third_slots_idx["M74"]),
            75: (winners_idx["F"], runners_idx["C"]),
            76: (winners_idx["C"], winners_idx["F"]),
            77: (winners_idx["I"], third_slots_idx["M77"]),
            78: (runners_idx["E"], runners_idx["I"]),
            79: (winners_idx["A"], third_slots_idx["M79"]),
            80: (winners_idx["L"], third_slots_idx["M80"]),
            81: (winners_idx["D"], third_slots_idx["M81"]),
            82: (winners_idx["G"], third_slots_idx["M82"]),
            83: (runners_idx["K"], runners_idx["L"]),
            84: (winners_idx["H"], winners_idx["J"]),
            85: (winners_idx["B"], third_slots_idx["M85"]),
            86: (winners_idx["J"], runners_idx["H"]),
            87: (winners_idx["K"], third_slots_idx["M87"]),
            88: (runners_idx["D"], runners_idx["G"]),
        }

        winners_by_match = {}
        for mid in range(73, 89):
            t1, t2 = r32[mid]
            winners_by_match[mid] = knockout_match_idx(rng, t1, t2)

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
            winners_by_match[mid] = knockout_match_idx(rng, t1, t2)

        qf = {
            97: (winners_by_match[89], winners_by_match[90]),
            98: (winners_by_match[93], winners_by_match[94]),
            99: (winners_by_match[91], winners_by_match[92]),
            100: (winners_by_match[95], winners_by_match[96]),
        }
        for mid in range(97, 101):
            t1, t2 = qf[mid]
            winners_by_match[mid] = knockout_match_idx(rng, t1, t2)

        sf = {
            101: (winners_by_match[97], winners_by_match[98]),
            102: (winners_by_match[99], winners_by_match[100]),
        }
        for mid in range(101, 103):
            t1, t2 = sf[mid]
            winners_by_match[mid] = knockout_match_idx(rng, t1, t2)

        champion = knockout_match_idx(rng, winners_by_match[101], winners_by_match[102])
        if champion == SCOTLAND_IDX:
            scotland_wins += 1

    return scotland_wins


if __name__ == "__main__":
    base = ITERATIONS // WORKERS
    remainder = ITERATIONS % WORKERS
    chunks = [(base + (1 if i < remainder else 0), i) for i in range(WORKERS)]
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        results = list(pool.map(run_chunk, chunks))
    print(sum(results))
