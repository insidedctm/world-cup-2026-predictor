# World Cup 2026 — Bayesian Group Stage Predictor

A technical reference for understanding, reproducing, and extending the model.

---

## Table of contents

1. [Background](#1-background)
2. [Repository structure](#2-repository-structure)
3. [Quick start — running the notebook in VS Code](#3-quick-start--running-the-notebook-in-vs-code)
4. [Data sources and preparation](#4-data-sources-and-preparation)
5. [The statistical model](#5-the-statistical-model)
6. [Prior construction](#6-prior-construction)
7. [Prediction methodology](#7-prediction-methodology)
8. [Results summary](#8-results-summary)
9. [Known limitations and caveats](#9-known-limitations-and-caveats)
10. [Extension ideas](#10-extension-ideas)
11. [Dependencies](#11-dependencies)

---

## 1. Background

This project adapts **Andrew Gelman's Bayesian hierarchical model** for the 2014 FIFA World Cup
([Attempt 1](https://statmodeling.stat.columbia.edu/2014/07/13/stan-analyzes-world-cup-data/),
[Attempt 2 / bug fix](https://statmodeling.stat.columbia.edu/2014/07/15/stan-world-cup-update/))
to predict the 72 group-stage matches of the 2026 FIFA World Cup.

The core idea: each team has a latent *ability* parameter. Score differences between teams are
observations of the difference in abilities, corrupted by noise. A Bayesian hierarchical model
allows teams with little data to borrow strength from the population-level prior (FIFA rankings),
while teams with many observations are pulled towards their empirical results.

**Key changes from the 2014 original:**

| Component | Gelman 2014 | This project |
|---|---|---|
| Tournament | 32-team (2014 WC) | 48-team (2026 WC) |
| Prior | FiveThirtyEight Soccer Power Index | FIFA Rankings (April 2025) |
| Training data | 2014 WC group games only | 2022 WC + 2026 qualifying |
| Output | Score difference distribution | Win / draw / loss percentages |
| Implementation | PyStan 2 | PyStan 3 (Stan 3 array syntax) |

---

## 2. Repository structure

After running the notebook end-to-end, the working directory contains:

```
wc2026/
├── data/
│   ├── training_matches.csv      # 110 match results used for fitting
│   ├── fixtures_2026.csv         # All 72 group-stage fixtures (raw)
│   ├── predictions.csv           # Final win/draw/loss predictions
│   ├── fifa_rankings_2025.json   # FIFA rankings for all 48 qualified teams
│   ├── a_samples.npy             # Posterior samples for team abilities (48 × 8000)
│   ├── sigma_y_samples.npy       # Posterior samples for σ_y (8000,)
│   ├── b_samples.npy             # Posterior samples for b (8000,)
│   ├── prior_score.npy           # Rescaled FIFA ranking scores (48,)
│   ├── teams.json                # Sorted list of 48 team names
│   └── team_idx.json             # team_name → 1-indexed position in teams list
└── notebooks/
    └── wc2026_predictions.ipynb  # Main notebook
```

**Inspectable data files:**

- `training_matches.csv` — raw match data with columns `date`, `home_team`, `away_team`,
  `home_score`, `away_score`, `source` (`wc2022` or `qual2026`).
- `predictions.csv` — one row per fixture with columns `group`, `date`, `team1`, `team2`,
  `team1_win_pct`, `draw_pct`, `team2_win_pct`, `sum_check`.
- `fifa_rankings_2025.json` — dictionary mapping team name to integer FIFA rank.

---

## 3. Quick start — running the notebook in VS Code

### Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.9 – 3.12 | 3.12 confirmed working |
| VS Code | Latest | |
| Jupyter extension | Latest | Microsoft `ms-toolsai.jupyter` |
| C++ compiler | — | Required to compile Stan models |

> **On macOS:** install Xcode Command Line Tools (`xcode-select --install`).  
> **On Windows:** install Visual Studio Build Tools with the "C++ build tools" workload.  
> **On Linux:** `sudo apt install build-essential` (Debian/Ubuntu).

### Step 1 — clone or download

If working from a git repository:

```bash
git clone <repo-url>
cd wc2026
```

Otherwise, place the notebook file `wc2026_predictions.ipynb` in a working directory.

### Step 2 — create a virtual environment

Creating an isolated environment avoids dependency conflicts, especially important given that
PyStan 3 and httpstan have specific version requirements.

```bash
# Create environment
python -m venv .venv

# Activate — macOS / Linux
source .venv/bin/activate

# Activate — Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
```

### Step 3 — install dependencies

```bash
pip install --upgrade pip

pip install pystan==3.10.0 \
            numpy \
            pandas \
            matplotlib \
            scipy \
            arviz
```

The most time-consuming install is `httpstan` (a dependency of `pystan`), which compiles
a C++ binary on first install. Expect 2–5 minutes on a typical machine.

> **Tip:** if you encounter a `RuntimeError` about Stan not compiling, ensure your C++
> compiler is on `PATH`. On macOS run `clang --version`; on Linux `gcc --version`.

### Step 4 — select the kernel in VS Code

1. Open the notebook file (`wc2026_predictions.ipynb`) in VS Code.
2. Click the kernel selector in the top-right corner of the notebook (it may show
   "Select Kernel" or a Python version).
3. Choose **Python Environments…** → select the `.venv` environment you just created.
   VS Code will show something like `Python 3.12.x ('.venv': venv)`.
4. If the `.venv` kernel does not appear, run:
   ```bash
   python -m ipykernel install --user --name=wc2026 --display-name "WC 2026"
   ```
   Then reopen the kernel picker and choose "WC 2026".

### Step 5 — run the notebook

Use **Run All** (`Shift+F5` or the ▶▶ button) to execute all cells in order, or run
cells individually with `Shift+Enter`.

**Expected runtimes:**

| Cell | Operation | Typical time |
|---|---|---|
| Data fetch | Download ~49k rows from GitHub | 3–10 seconds |
| Stan compile | Compile Stan model to C++ | 45–90 seconds (first run only) |
| MCMC sampling | 4 chains × 2000 samples | 30–120 seconds |
| Plotting | Generate all figures | 10–30 seconds |

Stan caches compiled models in a temporary directory. Subsequent runs in the same session
skip the compilation step.

### Troubleshooting

**`ModuleNotFoundError: No module named 'stan'`**
Ensure your `.venv` kernel is selected, not the system Python. The Jupyter extension
sometimes defaults to the system interpreter.

**Stan compilation fails on Windows**
Confirm Visual Studio Build Tools are installed and that `cl.exe` is on your `PATH`. An
easy check: open a Developer Command Prompt and type `cl`.

**`httpstan` install fails**
Try pinning the version: `pip install httpstan==4.13.0`. This is the version confirmed
compatible with `pystan==3.10.0`.

**`UnicodeDecodeError` reading CSV**
The dataset contains Unicode characters (notably `Curaçao`). Ensure you are using
`pd.read_csv(..., encoding='utf-8')`, which is the default in modern pandas.

**Kernel dies during sampling**
This usually indicates insufficient RAM. The model uses ~500 MB during sampling on 48
teams. Close other applications, or reduce `num_samples` to 1000 and `num_chains` to 2.

---

## 4. Data sources and preparation

### 4.1 Match results

All match data is fetched at runtime from the
[martj42/international_results](https://github.com/martj42/international_results)
GitHub repository. This is a community-maintained dataset of international football
results from 1872 to the present, updated regularly.

The fetch URL used in the notebook:
```
https://raw.githubusercontent.com/martj42/international_results/master/results.csv
```

Columns used: `date`, `home_team`, `away_team`, `home_score`, `away_score`, `tournament`.

### 4.2 Training set construction

Two subsets are extracted and concatenated:

**2022 FIFA World Cup** (`source = 'wc2022'`):
- Filter: `tournament == 'FIFA World Cup'` and `date` in `[2022-11-01, 2023-01-01)`.
- Further filter: both `home_team` and `away_team` in the set of 48 qualified nations.
- Result: **46 games**.

**2026 World Cup qualifying** (`source = 'qual2026'`):
- Filter: `tournament == 'FIFA World Cup qualification'` and `date > 2022-12-18`
  (after the Qatar 2022 final).
- Further filter: both teams among the 48 qualified nations, and `home_score` not null.
- Result: **64 games**.

**Combined training set: 110 games**, covering the period 2022-11-20 to 2025-11-18.

### 4.3 Why filter to qualified-team pairs only

Including games against non-qualifiers would inflate the estimated ability of strong
nations (e.g., Germany beating San Marino 10–0 says little about Germany's ability
relative to World Cup opponents). The bilateral filter ensures every training observation
is informative about the relative abilities of teams in the 2026 tournament.

**OFC qualifying was excluded** because Oceania produced only one qualifier (New Zealand)
via an intercontinental play-off. There are no competitive head-to-head qualifying games
between multiple OFC-qualified teams.

### 4.4 Coverage by team

The training set is unevenly distributed across teams:

| Games in training set | Teams |
|---|---|
| 0 (prior only) | Algeria, Cape Verde, Egypt, Ivory Coast, New Zealand, Norway, Panama, Scotland, South Africa |
| 1–3 (sparse) | Belgium, Curaçao, Canada, Germany, Haiti, Mexico, Sweden, Turkey, Czech Republic, Tunisia, Bosnia and Herzegovina, Austria, DR Congo, Ghana, United States |
| 4–16 (well-observed) | Argentina (16), Brazil (13), Uruguay (13), Ecuador (13), Colombia (10), Paraguay (10), South Korea (8), Iran (8), Croatia (9), Saudi Arabia (9), and others |

Teams with zero games have their ability estimates driven entirely by the FIFA ranking
prior. The Bayesian framework handles this gracefully: the posterior for these teams
simply equals their prior, and their uncertainty (posterior SD) is determined by `σ_a`.

### 4.5 2026 fixture list

The same `martj42` dataset includes the 72 scheduled 2026 group-stage fixtures with
`NaN` scores (since they have not yet been played). These are extracted with:

```python
fixtures = df_all[
    (df_all['tournament'] == 'FIFA World Cup') &
    (df_all['date'] >= '2026-06-01') &
    (df_all['home_team'].isin(qualified)) &
    (df_all['away_team'].isin(qualified))
]
```

The 2026 WC has **12 groups of 4 teams** (Groups A–L), each playing 6 games
(every team plays every other team once), giving 12 × 6 = **72 group-stage fixtures**.

---

## 5. The statistical model

### 5.1 Likelihood

For each training game *k*, let `y_k` be the score difference between the *favourite*
(Team 1, lower FIFA rank number) and the *underdog* (Team 2):

```
y_k = goals_favourite_k - goals_underdog_k
```

The likelihood is a Student-t distribution with fixed degrees of freedom ν = 7:

```
y_k ~ t_7( a[i(k)] - a[j(k)],  σ_y )
```

where `a[i]` is the latent ability of team `i` and `σ_y` is the observational noise.

The Student-t with ν = 7 has heavier tails than a normal distribution, accommodating
occasional dramatic scorelines (e.g., Germany 7–1 Brazil in 2014) without treating them
as statistical impossibilities. The degrees of freedom are fixed, not estimated, following
Gelman's original implementation.

### 5.2 Ability prior (hierarchical structure)

Each team's ability is drawn from a population-level distribution informed by the FIFA
ranking prior score `s_i`:

```
a_i ~ Normal( b * s_i,  σ_a )
```

The hyperparameter `b` controls how much weight is given to the FIFA rankings. If `b ≈ 0`,
the rankings carry no information and all teams are treated as equal a priori. In practice
the model estimated `b ≈ 1.24`, confirming that rankings are genuinely informative.

### 5.3 Non-centred parameterisation

To improve MCMC efficiency — particularly for teams with few observations where the
posterior geometry can be funnel-shaped — the model uses a non-centred parameterisation:

```
ã_i ~ Normal(0, 1)          # standardised latent ability
a_i = b * s_i + σ_a * ã_i   # actual ability
```

This is mathematically equivalent to the centred form but avoids the sampler getting
stuck in the narrow neck of the Neal's funnel.

### 5.4 Stan code

The full Stan model as used in the notebook:

```stan
data {
  int<lower=0> num_teams;
  int<lower=0> num_games;
  vector[num_teams] prior_score;
  array[num_games] int team_1_idx;   // 1-indexed, favourite
  array[num_games] int team_2_idx;   // 1-indexed, underdog
  vector[num_games] score_diff;
  real<lower=0> deg_freedom;         // fixed at 7.0
}
parameters {
  real b;
  real<lower=0> sigma_a;
  real<lower=0> sigma_y;
  vector[num_teams] raw_a;
}
transformed parameters {
  vector[num_teams] a;
  a = b * prior_score + sigma_a * raw_a;
}
model {
  raw_a ~ std_normal();
  for (i in 1:num_games)
    score_diff[i] ~ student_t(deg_freedom,
                              a[team_1_idx[i]] - a[team_2_idx[i]],
                              sigma_y);
}
generated quantities {
  vector[num_games] ypred_train;
  for (i in 1:num_games)
    ypred_train[i] = student_t_rng(deg_freedom,
                                   a[team_1_idx[i]] - a[team_2_idx[i]],
                                   sigma_y);
}
```

> **Note on Stan syntax:** This uses Stan 3 / PyStan 3 syntax. The key difference from the
> PyStan 2 version of Gelman's original is the use of `array[N] int` declarations in the
> `data` block. The older `int team_1_rank[num_games]` syntax is deprecated in Stan 2.26+
> and will cause a compilation warning or error in current versions.

### 5.5 Sampling configuration

```python
posterior = stan.build(stan_code, data=stan_data, random_seed=42)
fit = posterior.sample(num_chains=4, num_samples=2000, num_warmup=1000)
```

- **4 chains** run in parallel, each producing 2,000 post-warmup samples.
- **Total posterior draws:** 8,000 per parameter.
- **Sampler:** Hamiltonian Monte Carlo with No-U-Turn Sampler (NUTS), Stan's default.
- **Random seed:** 42, set for reproducibility.

### 5.6 Posterior summary

| Parameter | Posterior mean | Interpretation |
|---|---|---|
| `b` | ≈ 1.24 | FIFA rankings are informative; rankings explain ~1.24 units of ability per standardised ranking point |
| `σ_y` | ≈ 1.33 | Typical unexplained variation in score difference is ±1.3 goals given known abilities |
| `σ_a` | Estimated | Spread of team abilities around the prior prediction |

### 5.7 Calibration check

An in-sample posterior predictive check was performed: for each of the 110 training games,
the 95% posterior predictive interval for the score difference was computed. Only **3.6%**
of actual results fell outside the interval. A perfectly calibrated model would produce
exactly 5% — the slight under-coverage (3.6% vs 5%) means the intervals are marginally
conservative, which is a safe direction for prediction.

---

## 6. Prior construction

### 6.1 FIFA Rankings source

The prior uses the **official FIFA World Rankings as of April 2025**, the most recent
ranking update before the 2026 World Cup. These are hardcoded in the notebook as a Python
dictionary mapping team name to integer rank (lower = stronger):

```python
rankings = {
    "Argentina": 1,  "Spain": 2,     "France": 3,   "England": 4,
    "Brazil": 5,     "Belgium": 6,   "Netherlands": 7, "Portugal": 8,
    "Colombia": 9,   "Germany": 12,  "Croatia": 13,  "Uruguay": 14,
    # ... all 48 qualified teams
    "New Zealand": 95, "Haiti": 100,
}
```

The full ranking dictionary is saved to `data/fifa_rankings_2025.json` for inspection.

### 6.2 Rescaling

Raw ranks are converted to a continuous prior score using Gelman's two-standard-deviation
standardisation:

```python
inv_ranks   = max(ranks) - ranks + 1       # invert: rank 1 → highest score
prior_score = (inv_ranks - inv_ranks.mean()) / (2 * inv_ranks.std(ddof=1))
```

This produces scores on approximately [−1.5, +0.7], with:
- The average qualified team at score ≈ 0
- Argentina (rank 1) at ≈ +0.66
- Haiti (rank 100) at ≈ −1.54

### 6.3 Team indexing

Teams are sorted alphabetically and assigned a 1-based integer index (required by Stan's
array indexing). The mapping is stored in `data/team_idx.json`:

```python
teams    = sorted(rankings.keys())           # alphabetical
team_idx = {t: i+1 for i, t in enumerate(teams)}  # 1-indexed
```

### 6.4 Gelman convention — favourites as Team 1

When building the training arrays, the favourite (lower FIFA rank number = stronger) is
always assigned as Team 1, regardless of home/away status. The score difference is then:

```python
score_diff = goals_favourite - goals_underdog
```

A positive score difference means the favourite won by that many goals; negative means
an upset. This convention ensures consistent interpretability of the sign of `a_i - a_j`.

When reporting predictions for a fixture, the model tracks which listed team is the
favourite and maps back to Team 1 / Team 2 order for display.

---

## 7. Prediction methodology

### 7.1 Posterior predictive sampling

For each fixture with teams T1 and T2:

```python
def predict_match(team1, team2):
    # Determine favourite by FIFA rank
    t1r, t2r = rankings[team1], rankings[team2]
    fave, underdog, fave_is_t1 = (
        (team1, team2, True) if t1r <= t2r else (team2, team1, False)
    )

    # Index into posterior samples
    fi, ui = team_idx[fave] - 1, team_idx[underdog] - 1

    # Compute ability difference for each of the 8000 samples
    ability_diff = a_samples[fi, :] - a_samples[ui, :]

    # Sample from the posterior predictive distribution
    # (Student-t with 7 df, centred on ability_diff, scale sigma_y)
    ypred = t_dist.rvs(df=7, loc=ability_diff, scale=sigma_y)

    # Classify each sample
    fave_win  = np.mean(ypred >  0.5)
    draw      = np.mean((ypred >= -0.5) & (ypred <= 0.5))
    under_win = np.mean(ypred < -0.5)

    # Return in fixture order (team1 first)
    return (fave_win, draw, under_win) if fave_is_t1 else (under_win, draw, fave_win)
```

### 7.2 The draw threshold

The ±0.5 goal threshold is an approximation. Because the Student-t is a continuous
distribution, `P(exactly 0)` = 0 — draws would never occur without a threshold.
The ±0.5 threshold treats any predicted score difference in (−0.5, +0.5) as a draw,
on the basis that this corresponds to a true score difference of 0 (goals being integers).

The three probabilities always sum to exactly 1.0 by construction.

### 7.3 Monte Carlo precision

With 8,000 posterior samples, the Monte Carlo standard error on each probability is
approximately `sqrt(p*(1-p)/8000)`. For a 50% probability this is ≈ 0.56 percentage
points — sufficient for reporting to one decimal place.

---

## 8. Results summary

All 72 predictions are in `data/predictions.csv`. Selected highlights:

### Most lopsided fixtures

| Team 1 | Team 2 | T1 win% | Draw% | T2 win% |
|---|---|---|---|---|
| Brazil | Haiti | 89.1 | 6.9 | 4.0 |
| Morocco | Haiti | 87.5 | 8.1 | 4.4 |
| Belgium | New Zealand | 86.8 | 8.2 | 4.9 *(T2 win = Belgium)* |
| Iran | New Zealand | 81.2 | 11.9 | 6.9 |
| Germany | Curaçao | 80.2 | 12.2 | 7.7 |

### Most evenly matched fixtures

| Team 1 | Team 2 | T1 win% | Draw% | T2 win% |
|---|---|---|---|---|
| Ghana | Panama | 37.0 | 29.0 | 34.0 |
| Sweden | Tunisia | 34.6 | 27.3 | 38.1 |
| Colombia | Portugal | 34.4 | 27.1 | 38.4 |
| DR Congo | Uzbekistan | 38.9 | 27.1 | 34.1 |
| Turkey | Paraguay | 38.8 | 27.5 | 33.7 |

### Notable predictions

- **Brazil vs Morocco** (Group C): 41.7% / 27.2% / 31.2% — unusually competitive given Brazil's ranking; reflects Morocco's strong recent performances, including their 2022 WC semi-final run, which is captured in the training data.
- **Netherlands vs Japan** (Group F): 45.8% / 26.1% / 28.1% — close match reflecting Japan's strong qualifying campaign and 2022 WC performance (knockout stage, beating Spain and Germany).
- **Colombia vs Portugal** (Group K): 34.4% / 27.1% / 38.4% — Colombia rated higher than Portugal by the model despite Portugal's superior FIFA ranking, reflecting Colombia's heavy presence in the qualifying data (10 games vs Portugal's 5).

---

## 9. Known limitations and caveats

### 9.1 Draws are underestimated

The ±0.5 threshold on a continuous distribution produces draw probabilities in the
range 6–29%, whereas empirical World Cup group-stage draw rates are typically 22–28%.
The model tends to under-predict draws for closely-matched teams and over-predict them
implicitly for lopsided fixtures (where draws would be surprises in both cases).

**Fix:** Implement a bivariate Poisson model with a Dixon-Coles correction term for
low-scoring games (0–0, 1–0, 0–1, 1–1). This naturally handles the integer structure
of goals and the excess of draws.

### 9.2 Nine teams are prior-only

Algeria, Cape Verde, Egypt, Ivory Coast, New Zealand, Norway, Panama, Scotland, and
South Africa have no qualifying games against other qualified teams in the dataset.
Their ability estimates are pure expressions of FIFA ranking with no data update.

**Implication:** predictions for games involving these teams are essentially saying
"teams of these FIFA rankings typically produce these outcomes" — there is no
tournament-specific information for them beyond the ranking itself.

### 9.3 Prior and likelihood partially overlap

FIFA rankings are computed from recent international results, a subset of which appear
in the qualifying dataset. The prior and the likelihood are not fully independent —
strong qualifying performances are counted twice (once in the rankings, once in the data).

**Fix:** Use an independent prior such as club-level Elo ratings (e.g., from
[clubelo.com](http://clubelo.com) or [eloratings.net](https://www.eloratings.net)).
These are derived from club football and are largely uncorrelated with international
qualifying results.

### 9.4 No home advantage

Qualifying games include a home team, but the model treats all games as neutral.
The home team wins approximately 44% of international friendlies and qualifiers,
versus 35% for the away team. Ignoring this introduces modest bias into ability estimates
for teams whose qualifying record is predominantly home or away.

**Fix:** Add a binary home indicator to the training data and a home advantage
parameter `h` to the likelihood: `y_k ~ t_7(a[i(k)] - a[j(k)] + h * home_k, σ_y)`.
Set `home_k = 0` for 2022 WC games (neutral) and the 2026 fixtures.

### 9.5 No temporal decay

A qualifying result from September 2023 is weighted equally to one from November 2025.
Team compositions and form change significantly over a 2.5-year qualifying campaign.

**Fix:** Apply an exponential decay weight to each game:
```python
w_k = exp(-lambda * days_before_tournament_k)
```
Incorporate weights into Stan using `target += weight * student_t_lpdf(...)`.

### 9.6 Score difference vs goals

Modelling the score difference loses information about offensive and defensive quality
separately. A team that wins 3–2 looks identical to one that wins 1–0 in this model,
despite these suggesting very different playing styles and vulnerabilities.

**Fix:** Model home goals and away goals as independent Poisson variables, learning
separate attack and defence parameters per team (the Dixon-Coles or Maher model).

---

## 10. Extension ideas

The model is a clean starting point for more sophisticated approaches. Roughly ordered
from simple to complex:

### 10.1 Add home advantage parameter

Minimal change to the Stan model — one new parameter, one new data vector. Cleanly
separates home advantage from team ability in the qualifying data.

### 10.2 Temporal decay weighting

Weight older games less. Compute `days_to_tournament` for each training game and
add an `exp(-lambda * days/365)` weight. Lambda can be estimated from held-out data
or set by cross-validation.

### 10.3 Independent prior

Replace FIFA rankings with club-level Elo ratings, which are computed from club
football and largely independent of the qualifying results used as training data.
This eliminates the double-counting issue described in §9.3.

### 10.4 Bivariate Poisson / Dixon-Coles

Replace the Student-t on score differences with separate Poisson distributions for
each team's goals. The Dixon-Coles correction adds a term to inflate the probability
of 0–0, 1–0, 0–1, and 1–1 scorelines. This produces proper win/draw/loss probabilities
without needing a threshold, and also generates full predicted scorelines.

```stan
// Sketch of Poisson model structure
parameters {
  vector[num_teams] attack;
  vector[num_teams] defence;
  real home_adv;
}
model {
  for (i in 1:num_games) {
    real lambda_home = exp(attack[home[i]] - defence[away[i]] + home_adv * neutral[i]);
    real lambda_away = exp(attack[away[i]] - defence[home[i]]);
    goals_home[i] ~ poisson(lambda_home);
    goals_away[i] ~ poisson(lambda_away);
  }
}
```

### 10.5 Tournament simulation

Use the group-stage predictions to simulate full tournament outcomes. For each of
the 8,000 posterior samples:
1. Simulate all 72 group games by drawing from the posterior predictive.
2. Apply the 2026 group qualification rules (top 2 per group + 8 best third-placed).
3. Simulate knockout rounds through to the final.

This gives a full distribution over final tournament outcomes, including championship
probabilities for each team.

### 10.6 Elo-updated ability

Rather than using static FIFA rankings as a prior, fit a dynamic model where team
abilities evolve game-by-game according to an Elo-like update rule, with the
Bayesian model estimating the update rate alongside the other parameters.

### 10.7 Player-level features

The most significant extension: incorporate squad-level features (average club Elo
of squad members, key player availability, age profile) as team-level covariates.
This addresses the model's current blindness to squad changes and injuries.

---

## 11. Dependencies

### Python packages

| Package | Version tested | Purpose |
|---|---|---|
| `pystan` | 3.10.0 | Python interface to Stan |
| `httpstan` | 4.13.0 | Stan HTTP server (installed automatically with pystan) |
| `numpy` | ≥1.24 | Numerical arrays and operations |
| `pandas` | ≥1.5 | Data loading and manipulation |
| `matplotlib` | ≥3.7 | Plotting |
| `scipy` | ≥1.10 | `scipy.stats.t` for posterior predictive sampling |
| `arviz` | ≥0.15 | MCMC diagnostics (optional, used for trace plots) |

Install all at once:

```bash
pip install "pystan==3.10.0" numpy pandas matplotlib scipy arviz
```

### Stan

Stan is compiled and invoked automatically by PyStan — no separate installation is
needed. The `httpstan` package handles the Stan HTTP server. Stan itself is embedded
within the `httpstan` package.

**Confirmed working combination:**
- PyStan 3.10.0
- httpstan 4.13.0
- Stan (embedded) — uses 2.29+ array syntax
- Python 3.12

### Data

No local data files are needed before running the notebook. All data is fetched at
runtime from:

- Match results: `https://raw.githubusercontent.com/martj42/international_results/master/results.csv`
- FIFA rankings: hardcoded dictionary in the notebook (sourced from FIFA.com, April 2025)

The notebook writes all outputs to `wc2026/data/` on first run. Subsequent runs can
load from the saved `.npy` files to skip the Stan compilation and sampling steps.

---

*Model based on Gelman, A. (2014). "Stan analyzes World Cup data."
Statistical Modeling, Causal Inference, and Social Statistics blog.
Data sourced from martj42/international_results (GitHub).*
