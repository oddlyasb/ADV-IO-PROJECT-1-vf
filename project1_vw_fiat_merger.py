"""
Project 1 - Car demand and the VW-Fiat merger simulation.

Replication script. Requires `cars.csv` in the working directory.
All tables (CSV) and figures (PNG) are saved in ./output.

Models:
    ols : simple logit, log consumer price (ln_p), OLS
    iv  : logit, ln_p instrumented with ln_avexr, 2SLS
    rob : robustness of `iv`, price relative to income (princ) in levels, 2SLS
All models include controls (we, li, home) and brand, market and class fixed effects.
"""

import os

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from linearmodels.iv import IV2SLS
import pyblp
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import seaborn as sns

pyblp.options.digits = 3
pyblp.options.verbose = False

OUT_DIR = "output"
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------------
data = pd.read_csv("cars.csv")
assert (data.groupby(["ma", "ye"]).size() == 40).all()   # 40 cars per market


# ---------------------------------------------------------------------------
# 2. Country, year and average household size k_t (linear interpolation)
# ---------------------------------------------------------------------------
MA_TO_COUNTRY = {1: "Belgium", 2: "France", 3: "Germany", 4: "Italy", 5: "UK"}
data["country"] = data["ma"].map(MA_TO_COUNTRY)
data["year"] = data["ye"] + 1900

HH_SIZE_ANCHORS = {
    "Italy":   ([1971, 1981, 1991, 2001], [3.3, 3.0, 2.8, 2.5]),
    "France":  ([1968, 1975, 1990, 1999], [3.09, 2.9, 2.6, 2.42]),
    "Germany": ([1961, 1972, 2000, 2021], [2.9, 2.7, 2.2, 2.0]),
    "UK":      ([1971, 1991, 1998, 2011], [2.9, 2.48, 2.32, 2.35]),
    "Belgium": ([1991, 1995, 1998, 2005], [2.49, 2.43, 2.41, 2.32]),
}
years = np.arange(1970, 2000)
hh_size = pd.DataFrame(
    {c: interp1d(x, y, kind="linear", fill_value="extrapolate")(years)
     for c, (x, y) in HH_SIZE_ANCHORS.items()},
    index=years,
)
hh_size_long = (hh_size.stack().rename("k")
                       .rename_axis(["year", "country"]).reset_index())
data = data.merge(hh_size_long, on=["country", "year"], how="left", validate="m:1")

# market identifier: country-year
data["market"] = data["ma"].astype(str) + "_" + data["ye"].astype(str)


# ---------------------------------------------------------------------------
# 3. Market size M_t (number of households)
# ---------------------------------------------------------------------------
data["M_size"] = data["pop"] / data["k"]


# ---------------------------------------------------------------------------
# 4. Shares and mean utility delta_jt = ln s_jt - ln s_0t
# ---------------------------------------------------------------------------
data["Q"] = data.groupby(["ma", "ye"])["qu"].transform("sum")
assert (data["Q"] < data["M_size"]).all()                 # outside share > 0
data["s"] = data["qu"] / data["M_size"]
data["s0"] = 1 - data["Q"] / data["M_size"]
data["delta"] = np.log(data["s"]) - np.log(data["s0"])


# ---------------------------------------------------------------------------
# 5. Price variables and instrument
# ---------------------------------------------------------------------------
data["ln_p"] = np.log(data["pr"])                    # consumer price (VAT incl.)
data["p_prod"] = data["pr"] / (1 + data["tax"])      # producer price (VAT excl.)
data["ln_avexr"] = np.log(data["avexr"])             # exporter's exchange rate

PRICE_VAR = "ln_p"        # main models (ols, iv)
PRICE_VAR_ROB = "princ"   # robustness model (rob)
MODELS = ["ols", "iv", "rob"]
MODEL_NAMES = {"ols": "OLS", "iv": "2SLS", "rob": "Robustness (princ)"}
PRICE_VARS = {"ols": PRICE_VAR, "iv": PRICE_VAR, "rob": PRICE_VAR_ROB}


# ---------------------------------------------------------------------------
# 6-7. Demand estimation (OLS and 2SLS, heteroskedasticity-robust SE)
# ---------------------------------------------------------------------------
CONTROLS = "we + li + home"
FE = "C(brd) + C(market) + C(cla)"


def estimate(formula):
    # OLS if the formula has no [...] block, 2SLS otherwise
    return IV2SLS.from_formula(formula, data=data).fit()


def price_sensitivity(res, price_var):
    # a_j = -d delta_j / d ln p_j
    coef = res.params[price_var]
    if price_var.startswith("ln_"):
        return pd.Series(-coef, index=data.index)    # log price: a_j = alpha
    return -coef * data[price_var]                   # levels: a_j = alpha * price_j


results_lm = {
    "ols": estimate(f"delta ~ 1 + {PRICE_VAR} + {CONTROLS} + {FE}"),
    "iv":  estimate(f"delta ~ 1 + {CONTROLS} + {FE} + [{PRICE_VAR} ~ ln_avexr]"),
    "rob": estimate(f"delta ~ 1 + {CONTROLS} + {FE} + [{PRICE_VAR_ROB} ~ ln_avexr]"),
}

for m in MODELS:
    data[f"a_{m}"] = price_sensitivity(results_lm[m], PRICE_VARS[m])

# coefficient tables (fixed effects and intercept omitted)
coef_tables = []
for m in MODELS:
    r = results_lm[m]
    cols = [c for c in r.params.index if not c.startswith("C(") and c != "Intercept"]
    coef_tables.append(pd.DataFrame({
        "model": MODEL_NAMES[m], "variable": cols,
        "coef": r.params[cols].values, "se": r.std_errors[cols].values,
        "p_value": r.pvalues[cols].values,
    }))
pd.concat(coef_tables).to_csv(f"{OUT_DIR}/regression_coefficients.csv", index=False)

# price coefficient, fit and first-stage diagnostics
summary_rows = {}
for m in MODELS:
    r, var = results_lm[m], PRICE_VARS[m]
    ci = r.conf_int().loc[var]
    row = {"price_var": var, "coef": r.params[var], "se": r.std_errors[var],
           "ci_lower": ci["lower"], "ci_upper": ci["upper"],
           "N": int(r.nobs), "R2": r.rsquared,
           "first_stage_partial_R2": np.nan, "first_stage_F": np.nan}
    if m != "ols":
        n_instr = len(r.model.instruments.cols)
        diag = r.first_stage.diagnostics
        row["first_stage_partial_R2"] = diag["partial.rsquared"].iloc[0]
        row["first_stage_F"] = diag["f.stat"].iloc[0] / n_instr   # F = chi2 / q
    summary_rows[MODEL_NAMES[m]] = row
pd.DataFrame(summary_rows).T.to_csv(f"{OUT_DIR}/price_coefficient_summary.csv")


# ---------------------------------------------------------------------------
# 8. Elasticities (logit)
# ---------------------------------------------------------------------------
for m in MODELS:
    data[f"eps_own_{m}"] = -data[f"a_{m}"] * (1 - data["s"])
    data[f"eps_cross_{m}"] = data[f"a_{m}"] * data["s"]

(data.groupby("country")[[f"eps_own_{m}" for m in MODELS]]
     .agg(["mean", "min", "max"])
     .to_csv(f"{OUT_DIR}/own_elasticities_by_country.csv"))
(data.groupby("country")[[f"eps_cross_{m}" for m in MODELS]].mean()
     .to_csv(f"{OUT_DIR}/cross_elasticities_by_country.csv"))
(data.groupby("cla")[["eps_own_iv", "eps_own_rob"]].mean()
     .to_csv(f"{OUT_DIR}/own_elasticities_by_class.csv"))


def logit_elasticities(a, shares):
    # E[j, k] = d ln s_j / d ln p_k
    J = len(shares)
    E = np.tile(a * shares, (J, 1))               # cross: a_k * s_k
    np.fill_diagonal(E, -a * (1 - shares))        # own: -a_j * (1 - s_j)
    return E


def market_elasticity_matrix(m, country, year):
    mkt = (data[(data["country"] == country) & (data["year"] == year)]
           .sort_values(["brand", "s"], ascending=[True, False]))
    E = logit_elasticities(mkt[f"a_{m}"].to_numpy(), mkt["s"].to_numpy())
    return pd.DataFrame(E, index=mkt["type"], columns=mkt["type"]), mkt


def heatmap_vw_fiat(matrices, mkt, country, year, file_name):
    # VW and Fiat models only, common color scale centred at zero
    sel = mkt["brand"].isin(["VW", "fiat"]).to_numpy()
    sub = {name: el.loc[sel, sel] for name, el in matrices.items()}
    values = np.concatenate([e.to_numpy().ravel() for e in sub.values()])
    norm = TwoSlopeNorm(vmin=values.min(), vcenter=0, vmax=values.max())

    fig, axes = plt.subplots(1, len(sub), figsize=(8 * len(sub), 7))
    for i, (ax, (name, e)) in enumerate(zip(axes, sub.items())):
        sns.heatmap(e, cmap="RdBu", norm=norm, annot=True, fmt=".4f",
                    annot_kws={"size": 6}, ax=ax, cbar=(i == len(sub) - 1))
        ax.set_title(f"Own- and cross-price elasticities, {name} ({country} {year})")
        ax.set_xlabel("Model k (price change)")
        ax.set_ylabel("Model j (share response)")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/{file_name}", dpi=300)
    plt.close(fig)


EXAMPLE_COUNTRY, EXAMPLE_YEAR = "Italy", 1990
el_ols, example_mkt = market_elasticity_matrix("ols", EXAMPLE_COUNTRY, EXAMPLE_YEAR)
el_iv, _ = market_elasticity_matrix("iv", EXAMPLE_COUNTRY, EXAMPLE_YEAR)
el_rob, _ = market_elasticity_matrix("rob", EXAMPLE_COUNTRY, EXAMPLE_YEAR)

heatmap_vw_fiat({"OLS": el_ols, "2SLS": el_iv}, example_mkt,
                EXAMPLE_COUNTRY, EXAMPLE_YEAR,
                f"heatmap_elast_{EXAMPLE_COUNTRY}{EXAMPLE_YEAR}.png")
heatmap_vw_fiat({"2SLS (ln_p)": el_iv, "2SLS (princ)": el_rob}, example_mkt,
                EXAMPLE_COUNTRY, EXAMPLE_YEAR,
                f"heatmap_elast_rob_{EXAMPLE_COUNTRY}{EXAMPLE_YEAR}.png")


# ---------------------------------------------------------------------------
# 9. Diversion ratios between groups (logit: depend only on shares)
# ---------------------------------------------------------------------------
VW_CODE, FIAT_CODE = 26, 4
data["group"] = np.select(
    [data["frm"].eq(VW_CODE), data["frm"].eq(FIAT_CODE)],
    ["VW", "Fiat"],
    default="Others",
)

S = data.pivot_table(index=["country", "year"], columns="group",
                     values="s", aggfunc="sum", fill_value=0)
S["s0"] = 1 - S[["VW", "Fiat", "Others"]].sum(axis=1)
S["D_VW_Fiat"] = S["Fiat"] / (1 - S["VW"])
S["D_Fiat_VW"] = S["VW"] / (1 - S["Fiat"])
S["D_VW_outside"] = S["s0"] / (1 - S["VW"])
S["D_VW_Fiat_inside"] = S["D_VW_Fiat"] / (1 - S["D_VW_outside"])
S["D_Fiat_VW_inside"] = S["D_Fiat_VW"] / (1 - S["s0"] / (1 - S["Fiat"]))

(S.groupby("country")[["VW", "Fiat", "s0", "D_VW_Fiat", "D_Fiat_VW", "D_VW_outside",
                       "D_VW_Fiat_inside", "D_Fiat_VW_inside"]].mean()
  .to_csv(f"{OUT_DIR}/diversion_ratios_by_country.csv"))


# ---------------------------------------------------------------------------
# 10. Marginal costs from the multi-product Bertrand-Nash FOC
#     s + (Omega * Delta)(p - c) = 0,  p = producer price
# ---------------------------------------------------------------------------
def marginal_costs(mkt, m):
    s = mkt["s"].to_numpy()
    p = mkt["p_prod"].to_numpy()
    a = mkt[f"a_{m}"].to_numpy()
    firm = mkt["frm"].to_numpy()
    # Delta[j, k] = ds_k / dp_j = -(a_j / p_j) * s_j * (1{j=k} - s_k)
    Delta = (a / p)[:, None] * (np.outer(s, s) - np.diag(s))
    Omega = (firm[:, None] == firm[None, :]).astype(float)
    markup = np.linalg.solve(Omega * Delta, -s)
    return p - markup, markup / p


def compute_costs(m):
    parts = []
    for _, mkt in data.groupby("market"):
        c, lerner = marginal_costs(mkt, m)
        parts.append(pd.DataFrame({"c": c, "lerner": lerner}, index=mkt.index))
    tmp = pd.concat(parts)
    data[f"c_{m}"] = tmp["c"]
    data[f"lerner_{m}"] = tmp["lerner"]


for m in MODELS:
    compute_costs(m)

(data.groupby("country")[[f"lerner_{m}" for m in MODELS]].describe()
     .to_csv(f"{OUT_DIR}/lerner_by_country.csv"))
(data.groupby("group")[[f"lerner_{m}" for m in MODELS]].mean()
     .to_csv(f"{OUT_DIR}/lerner_by_group.csv"))
(data.assign(neg_cost=data["c_ols"] <= 0)
     .groupby(["country", "group"])["neg_cost"].sum()
     .to_csv(f"{OUT_DIR}/negative_costs_ols.csv"))


# ---------------------------------------------------------------------------
# 11-12. VW-Fiat merger simulation with PyBLP
# ---------------------------------------------------------------------------
product_data = pd.DataFrame({
    "market_ids": data["market"],
    "firm_ids":   data["frm"],          # pre-merger ownership
    "shares":     data["s"],
    "prices":     data["p_prod"],       # producer price, chosen by firms
    "we":   data["we"],
    "li":   data["li"],
    "home": data["home"],
    "brd":  data["brd"],
    "cla":  data["cla"],
}, index=data.index)
# princ = p_prod * (1 + tax) / (ngdp / pop): factor constant within a market
product_data["princ_factor"] = data["princ"] / data["p_prod"]

ABSORB = "C(brd) + C(market_ids) + C(cla)"
firm_post = data["frm"].replace({FIAT_CODE: VW_CODE}).to_numpy()   # Fiat -> VW


def simulate_merger(m, x1_formula, instrument):
    pdata = product_data.copy()
    pdata["demand_instruments0"] = instrument

    problem = pyblp.Problem(pyblp.Formulation(x1_formula, absorb=ABSORB), pdata)
    results = problem.solve(method="1s")
    # same price coefficient as linearmodels
    assert np.isclose(results.beta[0, 0], results_lm[m].params[PRICE_VARS[m]], rtol=1e-4)

    costs = results.compute_costs()                      # pre-merger costs
    p_post = results.compute_prices(
        firm_ids=firm_post, costs=costs,
        iteration=pyblp.Iteration("simple", {"atol": 1e-12, "max_evaluations": 10000}),
    )
    s_post = results.compute_shares(prices=p_post)

    data[f"p_post_{m}"] = p_post[:, 0]
    data[f"dp_{m}"] = p_post[:, 0] / data["p_prod"] - 1   # % price change
    data[f"s_post_{m}"] = s_post[:, 0]
    return results


# OLS: the instrument is the price itself (GMM = OLS)
results_blp = {
    "ols": simulate_merger("ols", "0 + log(prices) + we + li + home",
                           np.log(data["p_prod"])),
    "iv":  simulate_merger("iv", "0 + log(prices) + we + li + home",
                           data["ln_avexr"]),
    "rob": simulate_merger("rob", "0 + I(prices * princ_factor) + we + li + home",
                           data["ln_avexr"]),
}


def merger_table(m):
    # mean % price change after the merger, by country and group
    return (data.groupby(["country", "group"])[f"dp_{m}"].mean().unstack() * 100)[
        ["VW", "Fiat", "Others"]]


# change in the outside share (percentage points), by country
outside_change = {}
for m in MODELS:
    s0_post = 1 - data.groupby("market")[f"s_post_{m}"].transform("sum")
    outside_change[MODEL_NAMES[m]] = ((s0_post - data["s0"]) * 100).groupby(data["country"]).mean()
pd.DataFrame(outside_change).to_csv(f"{OUT_DIR}/outside_share_change.csv")


# ---------------------------------------------------------------------------
# 13. Final comparison of the three models
# ---------------------------------------------------------------------------
merger_comparison = pd.concat({MODEL_NAMES[m]: merger_table(m) for m in MODELS}, axis=1)
merger_comparison.to_csv(f"{OUT_DIR}/merger_price_changes.csv")

merged_products = data[data["group"].isin(["VW", "Fiat"])]
fig, axes = plt.subplots(1, len(MODELS), figsize=(6 * len(MODELS), 4))
for ax, m in zip(axes, MODELS):
    (merged_products.groupby(["year", "country"])[f"dp_{m}"].mean().unstack() * 100
     ).plot(ax=ax, marker="o", ms=3)
    ax.set_title(MODEL_NAMES[m])
    ax.set_xlabel("year")
    ax.set_ylabel("mean price change of VW and Fiat products (%)")
    ax.grid(alpha=.3)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/merger_by_market.png", dpi=300)
plt.close(fig)
