import numpy as np
import app

iterations = 50000
seed = 20260620
rng = np.random.default_rng(seed)
strength = app.build_strength_table()
lam_home, lam_away, elo_arr = app._precompute_goal_rate_matrices(strength)
locked_lookup_idx = {
    (g, app.TEAM_INDEX[h], app.TEAM_INDEX[a]): (gh, ga)
    for (g, h, a), (gh, ga) in app.merge_locked_results([]).items()
}

scotland_idx = app.TEAM_INDEX["Scotland"]
scotland_wins = 0

for _ in range(iterations):
    stats = np.zeros((len(app.ALL_TEAMS), 7), dtype=np.int16)

    winners_idx = {}
    runners_idx = {}
    third_rows = []

    for group in app.GROUP_ORDER:
        for home_idx, away_idx in app.GROUP_FIXTURE_INDICES[group]:
            key = (group, home_idx, away_idx)
            if key in locked_lookup_idx:
                gh, ga = locked_lookup_idx[key]
            else:
                gh = int(rng.poisson(lam_home[home_idx, away_idx]))
                ga = int(rng.poisson(lam_away[home_idx, away_idx]))
            app._apply_result_array(stats, home_idx, away_idx, gh, ga)

        sorted_group = app._sorted_group_indices(stats, app.GROUP_TEAM_INDICES[group], elo_arr)
        winners_idx[group] = int(sorted_group[0])
        runners_idx[group] = int(sorted_group[1])
        third_idx = int(sorted_group[2])
        third_rows.append(
            (
                group,
                third_idx,
                int(stats[third_idx, app.STAT_PTS]),
                int(stats[third_idx, app.STAT_GF] - stats[third_idx, app.STAT_GA]),
                int(stats[third_idx, app.STAT_GF]),
            )
        )

    third_rows_sorted = sorted(third_rows, key=lambda x: (x[2], x[3], x[4]), reverse=True)
    best_thirds = third_rows_sorted[:8]
    best_third_groups = [row[0] for row in best_thirds]
    best_third_indices = [row[1] for row in best_thirds]
    third_slots_idx = app._assign_third_place_slots_idx(best_third_groups, best_third_indices)

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
        winner = app._knockout_match_idx(t1, t2, lam_home, lam_away, elo_arr, rng)
        winners_by_match[mid] = winner
        losers_by_match[mid] = t2 if winner == t1 else t1

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
        winner = app._knockout_match_idx(t1, t2, lam_home, lam_away, elo_arr, rng)
        winners_by_match[mid] = winner
        losers_by_match[mid] = t2 if winner == t1 else t1

    qf = {
        97: (winners_by_match[89], winners_by_match[90]),
        98: (winners_by_match[93], winners_by_match[94]),
        99: (winners_by_match[91], winners_by_match[92]),
        100: (winners_by_match[95], winners_by_match[96]),
    }

    for mid in range(97, 101):
        t1, t2 = qf[mid]
        winner = app._knockout_match_idx(t1, t2, lam_home, lam_away, elo_arr, rng)
        winners_by_match[mid] = winner
        losers_by_match[mid] = t2 if winner == t1 else t1

    sf = {
        101: (winners_by_match[97], winners_by_match[98]),
        102: (winners_by_match[99], winners_by_match[100]),
    }

    for mid in range(101, 103):
        t1, t2 = sf[mid]
        winner = app._knockout_match_idx(t1, t2, lam_home, lam_away, elo_arr, rng)
        winners_by_match[mid] = winner
        losers_by_match[mid] = t2 if winner == t1 else t1

    champion = app._knockout_match_idx(winners_by_match[101], winners_by_match[102], lam_home, lam_away, elo_arr, rng)
    if champion == scotland_idx:
        scotland_wins += 1

print(scotland_wins)
