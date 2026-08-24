import argparse
import json
import math
import statistics
from itertools import combinations
from pathlib import Path

import numpy as np


# ============================================================
# Common-structure / "is it just 12 unrelated interactions?"
# control for the response-geometry experiment.
#
# NO MODEL INFERENCE.
# Uses only:
#   full_distribution_validation/selected_pairs.json
#   full_distribution_validation/profiles.jsonl
#
# Core test:
#   Build a 4-D response space for each intervention family:
#       repeat = [LUMA, NOVA, KITE, 4827]
#       copy   = [LUMA, NOVA, KITE, 4827]
#       write  = [LUMA, NOVA, KITE, 4827]
#
#   Then compare the 40-state pairwise geometry (780 distances)
#   across the three spaces, with state-label permutation tests.
# ============================================================

TASKS = ("repeat", "copy", "write")
PAYLOADS = ("luma", "nova", "kite", "4827")

CONDITIONS = {
    task: [f"{task}_{payload}" for payload in PAYLOADS]
    for task in TASKS
}


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if x.size < 2:
        return float("nan")

    x = x - x.mean()
    y = y - y.mean()

    den = math.sqrt(float(np.dot(x, x) * np.dot(y, y)))
    if den == 0:
        return float("nan")

    return float(np.dot(x, y) / den)


def rankdata_average(x):
    """Average ranks for ties, 0-based ranks. No scipy required."""
    x = np.asarray(x)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)

    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1

        avg_rank = (i + j - 1) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j

    return ranks


def spearman(x, y):
    return pearson(rankdata_average(x), rankdata_average(y))


def pairwise_mae_matrix(X):
    """
    X: [n_states, n_features]
    D[i,j] = mean absolute coordinate difference.
    """
    X = np.asarray(X, dtype=np.float64)
    return np.abs(X[:, None, :] - X[None, :, :]).mean(axis=2)


def upper_triangle(D):
    idx = np.triu_indices(D.shape[0], k=1)
    return D[idx]


def permute_matrix(D, perm):
    return D[np.ix_(perm, perm)]


def rsa_permutation_test(Da, Db, n_perm, rng):
    """
    Representational Similarity Analysis:
    correlation between all upper-triangle state-state distances.

    Null: state identities in Db are unrelated to Da.
    One-sided p because the common-geometry hypothesis predicts positive r.
    """
    va = upper_triangle(Da)
    vb = upper_triangle(Db)

    obs_p = pearson(va, vb)
    obs_s = spearman(va, vb)

    null_p = np.empty(n_perm, dtype=np.float64)
    null_s = np.empty(n_perm, dtype=np.float64)

    n = Da.shape[0]

    for k in range(n_perm):
        perm = rng.permutation(n)
        Dp = permute_matrix(Db, perm)
        vp = upper_triangle(Dp)

        null_p[k] = pearson(va, vp)
        null_s[k] = spearman(va, vp)

    # +1 correction gives an exact-style Monte Carlo p estimate.
    pval_p = (1 + np.sum(null_p >= obs_p)) / (n_perm + 1)
    pval_s = (1 + np.sum(null_s >= obs_s)) / (n_perm + 1)

    null_p_mean = float(null_p.mean())
    null_p_sd = float(null_p.std(ddof=1))

    z_p = (
        (obs_p - null_p_mean) / null_p_sd
        if null_p_sd > 0
        else float("inf")
    )

    return {
        "pearson_r": obs_p,
        "spearman_rho": obs_s,
        "pearson_perm_p_one_sided": float(pval_p),
        "spearman_perm_p_one_sided": float(pval_s),
        "pearson_null_mean": null_p_mean,
        "pearson_null_sd": null_p_sd,
        "pearson_z_vs_null": float(z_p),
    }


def susceptibility(X):
    """
    One scalar per state: mean absolute response magnitude in that family.
    Tests whether some states are globally 'more sensitive' across families.
    """
    X = np.asarray(X, dtype=np.float64)
    return np.abs(X).mean(axis=1)


def susceptibility_permutation_test(a, b, n_perm, rng):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    obs_p = pearson(a, b)
    obs_s = spearman(a, b)

    null_p = np.empty(n_perm, dtype=np.float64)
    null_s = np.empty(n_perm, dtype=np.float64)

    for k in range(n_perm):
        bp = b[rng.permutation(len(b))]
        null_p[k] = pearson(a, bp)
        null_s[k] = spearman(a, bp)

    pval_p = (1 + np.sum(null_p >= obs_p)) / (n_perm + 1)
    pval_s = (1 + np.sum(null_s >= obs_s)) / (n_perm + 1)

    return {
        "pearson_r": obs_p,
        "spearman_rho": obs_s,
        "pearson_perm_p_one_sided": float(pval_p),
        "spearman_perm_p_one_sided": float(pval_s),
    }


def nearest_neighbor_transfer(D_source, D_target):
    """
    For each state:
      - find its nearest other state in source geometry
      - ask what rank that same state has in target geometry

    Rank 1 = also nearest in target.
    Normalized percentile:
      0 = best possible transfer
      1 = worst possible transfer
    """
    n = D_source.shape[0]

    target_ranks = []
    percentiles = []

    for i in range(n):
        source_candidates = [
            j for j in range(n) if j != i
        ]
        nearest = min(
            source_candidates,
            key=lambda j: D_source[i, j]
        )

        target_candidates = sorted(
            source_candidates,
            key=lambda j: D_target[i, j]
        )

        rank = target_candidates.index(nearest) + 1
        target_ranks.append(rank)

        if n > 2:
            pct = (rank - 1) / (n - 2)
        else:
            pct = 0.0
        percentiles.append(pct)

    return {
        "median_target_rank": float(statistics.median(target_ranks)),
        "mean_target_rank": float(statistics.fmean(target_ranks)),
        "median_normalized_percentile": float(statistics.median(percentiles)),
        "mean_normalized_percentile": float(statistics.fmean(percentiles)),
        "fraction_top5": float(
            sum(r <= 5 for r in target_ranks) / len(target_ranks)
        ),
        "fraction_top10": float(
            sum(r <= 10 for r in target_ranks) / len(target_ranks)
        ),
    }


def matched_pair_family_distances(pairs, state_index, D_by_task):
    rows = []

    for pi, pair in enumerate(pairs, 1):
        ha = pair["a"]["order_hash"]
        hb = pair["b"]["order_hash"]

        ia = state_index[ha]
        ib = state_index[hb]

        row = {
            "pair": pi,
            "family": pair["a"]["family"],
            "a_hash": ha,
            "b_hash": hb,
        }

        for task in TASKS:
            row[f"{task}_distance"] = float(
                D_by_task[task][ia, ib]
            )

        rows.append(row)

    return rows


def bootstrap_corr_ci(x, y, n_boot, rng):
    """
    Simple nonparametric bootstrap over the 20 selected matched pairs.
    Secondary descriptive analysis only.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    obs = pearson(x, y)

    vals = []
    n = len(x)

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)

        xb = x[idx]
        yb = y[idx]

        r = pearson(xb, yb)
        if not math.isnan(r):
            vals.append(r)

    if not vals:
        return {
            "r": obs,
            "bootstrap_95_ci": [float("nan"), float("nan")],
        }

    lo, hi = np.quantile(vals, [0.025, 0.975])

    return {
        "r": obs,
        "bootstrap_95_ci": [float(lo), float(hi)],
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--pairs",
        default="full_distribution_validation/selected_pairs.json",
    )
    ap.add_argument(
        "--profiles",
        default="full_distribution_validation/profiles.jsonl",
    )
    ap.add_argument(
        "--permutations",
        type=int,
        default=10000,
    )
    ap.add_argument(
        "--bootstrap",
        type=int,
        default=10000,
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=20260824,
    )
    ap.add_argument(
        "--out",
        default="geometry_structure_summary.json",
    )

    args = ap.parse_args()

    pairs_path = Path(args.pairs)
    profiles_path = Path(args.profiles)

    if not pairs_path.exists():
        raise FileNotFoundError(
            f"Missing {pairs_path}\n"
            "Run this script from the same directory as "
            "full_distribution_validation/, or pass --pairs."
        )

    if not profiles_path.exists():
        raise FileNotFoundError(
            f"Missing {profiles_path}\n"
            "Run this script from the same directory as "
            "full_distribution_validation/, or pass --profiles."
        )

    with open(pairs_path, "r", encoding="utf-8") as f:
        pair_data = json.load(f)

    pairs = pair_data["pairs"]
    profile_rows = load_jsonl(profiles_path)

    # --------------------------------------------------------
    # Recover the 40 selected states and their baseline LO.
    # --------------------------------------------------------

    baseline_lo = {}
    family_of = {}

    for pair in pairs:
        for side in ("a", "b"):
            r = pair[side]
            h = r["order_hash"]

            baseline_lo[h] = float(r["logodds"])
            family_of[h] = r["family"]

    state_hashes = sorted(baseline_lo)
    state_index = {
        h: i for i, h in enumerate(state_hashes)
    }

    n_states = len(state_hashes)

    # --------------------------------------------------------
    # Build delta-log-odds response lookup.
    # --------------------------------------------------------

    raw_lo = {}

    selected_hashes = set(state_hashes)

    for row in profile_rows:
        h = row["order_hash"]
        cond = row["condition"]

        if h in selected_hashes:
            raw_lo[(h, cond)] = float(row["logodds"])

    missing = []

    for h in state_hashes:
        for task in TASKS:
            for cond in CONDITIONS[task]:
                if (h, cond) not in raw_lo:
                    missing.append((h, cond))

    if missing:
        preview = "\n".join(
            f"  {h} {cond}"
            for h, cond in missing[:20]
        )
        raise RuntimeError(
            f"Missing {len(missing)} selected-state profile rows.\n"
            f"First missing rows:\n{preview}"
        )

    # X_by_task[task] = [40 states, 4 payloads]
    X_by_task = {}

    for task in TASKS:
        X = np.zeros(
            (n_states, len(PAYLOADS)),
            dtype=np.float64,
        )

        for i, h in enumerate(state_hashes):
            base = baseline_lo[h]

            for j, cond in enumerate(CONDITIONS[task]):
                X[i, j] = raw_lo[(h, cond)] - base

        X_by_task[task] = X

    # --------------------------------------------------------
    # Geometry: all 780 pairwise state-state distances.
    # --------------------------------------------------------

    D_by_task = {
        task: pairwise_mae_matrix(X_by_task[task])
        for task in TASKS
    }

    rng = np.random.default_rng(args.seed)

    family_pairs = [
        ("repeat", "copy"),
        ("repeat", "write"),
        ("copy", "write"),
    ]

    print("=" * 92)
    print("COMMON-STRUCTURE TEST")
    print("=" * 92)
    print(f"States: {n_states}")
    print(
        f"State-state distances per geometry: "
        f"{n_states * (n_states - 1) // 2}"
    )
    print(f"Permutation iterations: {args.permutations}")
    print()

    rsa_results = {}

    print("1) CROSS-FAMILY GEOMETRY (RSA)")
    print("   Do the same state pairs stay near/far across repeat/copy/write?")
    print()

    for a, b in family_pairs:
        res = rsa_permutation_test(
            D_by_task[a],
            D_by_task[b],
            args.permutations,
            rng,
        )

        key = f"{a}_vs_{b}"
        rsa_results[key] = res

        print(
            f"  {a:>6} vs {b:<6} | "
            f"Pearson r={res['pearson_r']:+.4f}  "
            f"Spearman ρ={res['spearman_rho']:+.4f}  "
            f"perm-p(r)={res['pearson_perm_p_one_sided']:.6g}  "
            f"z={res['pearson_z_vs_null']:+.2f}"
        )

    # --------------------------------------------------------
    # Per-state susceptibility.
    # --------------------------------------------------------

    print()
    print("2) PER-STATE SUSCEPTIBILITY")
    print("   Are states that react strongly in one family also strong in another?")
    print()

    susc = {
        task: susceptibility(X_by_task[task])
        for task in TASKS
    }

    susceptibility_results = {}

    for a, b in family_pairs:
        res = susceptibility_permutation_test(
            susc[a],
            susc[b],
            args.permutations,
            rng,
        )

        key = f"{a}_vs_{b}"
        susceptibility_results[key] = res

        print(
            f"  {a:>6} vs {b:<6} | "
            f"Pearson r={res['pearson_r']:+.4f}  "
            f"Spearman ρ={res['spearman_rho']:+.4f}  "
            f"perm-p(r)={res['pearson_perm_p_one_sided']:.6g}"
        )

    # --------------------------------------------------------
    # Nearest-neighbor transfer.
    # --------------------------------------------------------

    print()
    print("3) LOCAL NEIGHBOR TRANSFER")
    print(
        "   If B is A's nearest state in one geometry, "
        "is B also nearby in another?"
    )
    print(
        "   Random expectation: median target rank ~20 of 39; "
        "normalized percentile ~0.5."
    )
    print()

    nn_results = {}

    for source in TASKS:
        for target in TASKS:
            if source == target:
                continue

            res = nearest_neighbor_transfer(
                D_by_task[source],
                D_by_task[target],
            )

            key = f"{source}_to_{target}"
            nn_results[key] = res

            print(
                f"  {source:>6} -> {target:<6} | "
                f"median rank={res['median_target_rank']:.1f}/39  "
                f"median pct={res['median_normalized_percentile']:.3f}  "
                f"top5={res['fraction_top5']:.3f}"
            )

    # --------------------------------------------------------
    # Secondary: only the 20 originally selected matched pairs.
    # --------------------------------------------------------

    matched_rows = matched_pair_family_distances(
        pairs,
        state_index,
        D_by_task,
    )

    print()
    print("4) ORIGINAL 20 MATCHED PAIRS (secondary)")
    print(
        "   Do pair-level separation magnitudes co-vary across families?"
    )
    print()

    matched_pair_corrs = {}

    for a, b in family_pairs:
        xa = [
            row[f"{a}_distance"]
            for row in matched_rows
        ]
        xb = [
            row[f"{b}_distance"]
            for row in matched_rows
        ]

        res = bootstrap_corr_ci(
            xa,
            xb,
            args.bootstrap,
            rng,
        )

        key = f"{a}_vs_{b}"
        matched_pair_corrs[key] = res

        lo, hi = res["bootstrap_95_ci"]

        print(
            f"  {a:>6} vs {b:<6} | "
            f"r={res['r']:+.4f}  "
            f"bootstrap 95% CI=[{lo:+.4f}, {hi:+.4f}]"
        )

    # --------------------------------------------------------
    # Guardrail interpretation.
    # --------------------------------------------------------

    rsa_rs = [
        rsa_results[k]["pearson_r"]
        for k in rsa_results
    ]

    rsa_ps = [
        rsa_results[k]["pearson_perm_p_one_sided"]
        for k in rsa_results
    ]

    strong = sum(
        (r >= 0.40 and p <= 0.01)
        for r, p in zip(rsa_rs, rsa_ps)
    )

    weak = sum(
        (abs(r) < 0.20 and p > 0.05)
        for r, p in zip(rsa_rs, rsa_ps)
    )

    print()
    print("=" * 92)
    print("INTERPRETATION GUARDRAIL")
    print("=" * 92)

    if strong >= 2:
        verdict = (
            "COMMON_STRUCTURE_SIGNAL"
        )
        print(
            "At least two of the three intervention-family geometries "
            "show strong positive cross-family structure under label permutation."
        )
        print(
            "The '12 unrelated idiosyncratic interactions' explanation "
            "takes a serious hit."
        )
    elif weak == 3:
        verdict = (
            "IDIOSYNCRATIC_EXPLANATION_SURVIVES"
        )
        print(
            "All three cross-family geometry correlations are weak and "
            "non-significant."
        )
        print(
            "The simplest reading is perturbation-family-specific interaction, "
            "not a shared local geometry."
        )
    else:
        verdict = "MIXED"
        print(
            "The result is mixed: there is some cross-family structure, "
            "but not enough for a clean shared-geometry claim."
        )
        print(
            "Inspect RSA, susceptibility, and neighbor-transfer together "
            "before deciding the next experiment."
        )

    # --------------------------------------------------------
    # Save complete result.
    # --------------------------------------------------------

    summary = {
        "n_states": n_states,
        "n_pairwise_distances": (
            n_states * (n_states - 1) // 2
        ),
        "tasks": list(TASKS),
        "payloads": list(PAYLOADS),
        "permutations": args.permutations,
        "seed": args.seed,
        "rsa_cross_family_geometry": rsa_results,
        "per_state_susceptibility": susceptibility_results,
        "nearest_neighbor_transfer": nn_results,
        "matched_pair_secondary": {
            "correlations": matched_pair_corrs,
            "rows": matched_rows,
        },
        "verdict_guardrail": verdict,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
