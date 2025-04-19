import numpy as np
import pandas as pd

# series_probs[i, j] = P(seed i beats seed j), 1-indexed: 1 through 8
series_probs = np.ones((9, 9)) / 7
hist_dist = [0.4489795918, 0.2551020408, 0.112244898, 0.05102040816, 0.04081632653, 0.0306122449, 0.02040816327, 0.04081632653]
seeds = np.arange(1, 9)

def simulate_one_season(series_probs):
    # Simulate one conference and then pick other finalist from historical dist
    # Round 1 matchups: (1 vs 8), (2 vs 7), (3 vs 6), (4 vs 5)
    winners = {}
    for (i, j) in [(1,8), (2,7), (3,6), (4,5)]:
        p = series_probs[i, j]
        # True means seed i advances; False means seed j advances
        winners[(i,j)] = np.random.rand() < p
        
    # If seed 8 wins, its next opponent is the winner of (4 vs 5)
    if winners[(1,8)]:
        return False
    opp_seed = 5 if winners[(4,5)] else 4
    p = series_probs[8, opp_seed]
    if np.random.rand() >= p:
        return False
    # Round 3 (Finals): opponent is winner of the other half
    # Approximate its seed: average seed among possible winners
    sampled_seed = np.random.choice(seeds, p=hist_dist)
    p_final = series_probs[8, sampled_seed]
    return np.random.rand() < p_final

def run_simulation(n_years=80, sims_per_year=10000, start_year=2025):
    results = []
    for year in range(start_year, start_year + n_years):
        for _ in range(sims_per_year):
            if simulate_one_season(series_probs):
                results.append(year)
        print(f"Year {year}: detected {sum([r==year for r in results])} wins")
    return pd.Series(results, name='winning_year')

dist = run_simulation(n_years=80, sims_per_year=100, start_year=2025)
