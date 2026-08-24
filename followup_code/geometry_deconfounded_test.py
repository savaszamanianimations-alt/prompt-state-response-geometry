import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np


TASKS = ("repeat", "copy", "write")
PAYLOADS = ("luma", "nova", "kite", "4827")
CONDS = {t: [f"{t}_{p}" for p in PAYLOADS] for t in TASKS}


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
    if len(x) < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    den = math.sqrt(float(np.dot(x, x) * np.dot(y, y)))
    if den == 0:
        return float("nan")
    return float(np.dot(x, y) / den)


def rankdata(x):
    x = np.asarray(x)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        r[order[i:j]] = (i + j - 1) / 2.0
        i = j
    return r


def spearman(x, y):
    return pearson(rankdata(x), rankdata(y))


def mae_matrix(X):
    X = np.asarray(X, dtype=np.float64)
    return np.abs(X[:, None, :] - X[None, :, :]).mean(axis=2)


def residualize(y, controls):
    y = np.asarray(y, dtype=np.float64)
    X = np.asarray(controls, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    X = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def partial_corr(x, y, controls):
    return pearson(residualize(x, controls), residualize(y, controls))


def within_family_pairs(indices_by_family):
    rows = []
    for fam, idxs in sorted(indices_by_family.items()):
        for ai in range(len(idxs)):
            for bi in range(ai + 1, len(idxs)):
                rows.append((fam, idxs[ai], idxs[bi]))
    return rows


def family_zscore(values, families):
    values = np.asarray(values, dtype=np.float64)
    out = np.empty_like(values)
    fams = np.asarray(families, dtype=object)

    for fam in sorted(set(families)):
        mask = fams == fam
        v = values[mask]
        sd = v.std(ddof=0)
        out[mask] = (v - v.mean()) / sd if sd > 0 else 0.0

    return out


def perm_test_corr_within_family(x, y, family_blocks, n_perm, rng):
    """
    Permute y observations only within family blocks.
    Used for pooled within-family pair-level quantities.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    family_blocks = np.asarray(family_blocks, dtype=object)

    obs = pearson(x, y)
    null = np.empty(n_perm, dtype=np.float64)

    uniq = sorted(set(family_blocks.tolist()))
    masks = [np.where(family_blocks == fam)[0] for fam in uniq]

    for k in range(n_perm):
        yp = y.copy()
        for idx in masks:
            yp[idx] = y[rng.permutation(idx)]
        null[k] = pearson(x, yp)

    p = (1 + np.sum(null >= obs)) / (n_perm + 1)
    sd = null.std(ddof=1)
    z = (obs - null.mean()) / sd if sd > 0 else float("inf")
    return {
        "r": float(obs),
        "p_one_sided": float(p),
        "null_mean": float(null.mean()),
        "null_sd": float(sd),
        "z": float(z),
    }


def state_label_perm_rsa_within_families(Da, Db, indices_by_family, n_perm, rng):
    """
    RSA over ONLY within-family state pairs.
    Null permutes state identities in Db independently within each family.
    This preserves family membership and family-scale structure.
    """
    pairs = within_family_pairs(indices_by_family)

    va = np.array([Da[i, j] for _, i, j in pairs], dtype=np.float64)
    vb = np.array([Db[i, j] for _, i, j in pairs], dtype=np.float64)
    fams = np.array([fam for fam, _, _ in pairs], dtype=object)

    # Z-score distances within family before pooling, so a "sensitive family"
    # cannot create the cross-task correlation by itself.
    za = family_zscore(va, fams)
    zb = family_zscore(vb, fams)

    obs = pearson(za, zb)
    obs_s = spearman(za, zb)

    null = np.empty(n_perm, dtype=np.float64)
    null_s = np.empty(n_perm, dtype=np.float64)

    # Keep Da fixed; relabel Db states inside each family.
    family_idx_arrays = {
        fam: np.array(idxs, dtype=int)
        for fam, idxs in indices_by_family.items()
    }

    for k in range(n_perm):
        mapping = np.arange(Da.shape[0])

        for fam, idxs in family_idx_arrays.items():
            mapping[idxs] = rng.permutation(idxs)

        vb_p = np.array(
            [Db[mapping[i], mapping[j]] for _, i, j in pairs],
            dtype=np.float64
        )
        zb_p = family_zscore(vb_p, fams)

        null[k] = pearson(za, zb_p)
        null_s[k] = spearman(za, zb_p)

    p = (1 + np.sum(null >= obs)) / (n_perm + 1)
    ps = (1 + np.sum(null_s >= obs_s)) / (n_perm + 1)

    sd = null.std(ddof=1)
    z = (obs - null.mean()) / sd if sd > 0 else float("inf")

    return {
        "n_within_family_distances": len(pairs),
        "pearson_r_family_zscored": float(obs),
        "spearman_rho_family_zscored": float(obs_s),
        "perm_p_r": float(p),
        "perm_p_rho": float(ps),
        "null_mean": float(null.mean()),
        "null_sd": float(sd),
        "z": float(z),
        "raw_a": va,
        "raw_b": vb,
        "families": fams,
        "pairs": pairs,
    }


def bootstrap_matched_pairs_family_cluster(x, y, fams, n_boot, rng):
    """
    Cluster bootstrap over the 5 evidence families.
    Resamples families, then includes all 4 selected pairs from each sampled family.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    fams = np.asarray(fams, dtype=object)
    unique = sorted(set(fams.tolist()))

    obs = pearson(
        family_zscore(x, fams),
        family_zscore(y, fams),
    )

    vals = []
    for _ in range(n_boot):
        sampled_fams = rng.choice(unique, size=len(unique), replace=True)
        xb = []
        yb = []
        fb = []

        # Give repeated sampled clusters unique labels so z-scoring is per draw.
        for draw_i, fam in enumerate(sampled_fams):
            idx = np.where(fams == fam)[0]
            xb.extend(x[idx].tolist())
            yb.extend(y[idx].tolist())
            fb.extend([f"{fam}__draw{draw_i}"] * len(idx))

        xb = np.asarray(xb)
        yb = np.asarray(yb)
        fb = np.asarray(fb, dtype=object)

        r = pearson(
            family_zscore(xb, fb),
            family_zscore(yb, fb),
        )
        if not math.isnan(r):
            vals.append(r)

    lo, hi = np.quantile(vals, [0.025, 0.975])
    return {
        "within_family_demeaned_r": float(obs),
        "cluster_bootstrap_95_ci": [float(lo), float(hi)],
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
    ap.add_argument("--permutations", type=int, default=10000)
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260824)
    ap.add_argument(
        "--out",
        default="geometry_deconfounded_summary.json",
    )
    args = ap.parse_args()

    with open(args.pairs, "r", encoding="utf-8") as f:
        pair_data = json.load(f)
    pairs = pair_data["pairs"]
    profiles = load_jsonl(args.profiles)

    # Recover 40 selected states.
    baseline = {}
    family = {}

    for pair in pairs:
        for side in ("a", "b"):
            r = pair[side]
            h = r["order_hash"]
            baseline[h] = float(r["logodds"])
            family[h] = r["family"]

    hashes = sorted(baseline)
    idx = {h: i for i, h in enumerate(hashes)}
    n = len(hashes)

    indices_by_family = defaultdict(list)
    for h in hashes:
        indices_by_family[family[h]].append(idx[h])

    raw_lo = {}
    for r in profiles:
        h = r["order_hash"]
        if h in idx:
            raw_lo[(h, r["condition"])] = float(r["logodds"])

    # Delta-response matrix: task -> state x payload.
    X = {}
    # Raw perturbed LO matrix: task -> state x payload.
    L = {}

    for task in TASKS:
        dx = np.zeros((n, 4), dtype=np.float64)
        lx = np.zeros((n, 4), dtype=np.float64)

        for i, h in enumerate(hashes):
            for j, cond in enumerate(CONDS[task]):
                if (h, cond) not in raw_lo:
                    raise RuntimeError(f"Missing profile row: {h} {cond}")
                val = raw_lo[(h, cond)]
                lx[i, j] = val
                dx[i, j] = val - baseline[h]

        X[task] = dx
        L[task] = lx

    # Geometry variants.
    D_delta = {t: mae_matrix(X[t]) for t in TASKS}

    # Center each state's 4 payload coordinates.
    # This EXACTLY removes the shared -baseline[state] offset.
    X_centered = {
        t: X[t] - X[t].mean(axis=1, keepdims=True)
        for t in TASKS
    }
    D_centered = {t: mae_matrix(X_centered[t]) for t in TASKS}

    # Raw perturbed-logodds geometry (no baseline subtraction at all).
    D_raw = {t: mae_matrix(L[t]) for t in TASKS}

    rng = np.random.default_rng(args.seed)
    task_pairs = [("repeat", "copy"), ("repeat", "write"), ("copy", "write")]

    print("=" * 96)
    print("DECONFOUNDED COMMON-STRUCTURE TEST")
    print("=" * 96)
    print(f"States: {n}  |  families: {len(indices_by_family)}")
    print("Each family contributes 8 selected states -> 28 within-family distances.")
    print(f"Total within-family distances: {sum(len(v)*(len(v)-1)//2 for v in indices_by_family.values())}")
    print()

    results = {
        "within_family_delta_geometry": {},
        "within_family_centered_geometry": {},
        "within_family_raw_perturbed_geometry": {},
        "partial_corr_controlling_baseline_distance": {},
        "matched_pairs_family_deconfounded": {},
    }

    print("1) WITHIN-FAMILY RSA, FAMILY SCALE REMOVED")
    print("   Uses only 140 within-family distances and z-scores inside each family.")
    print()

    delta_rsa_cache = {}

    for a, b in task_pairs:
        res = state_label_perm_rsa_within_families(
            D_delta[a], D_delta[b],
            indices_by_family, args.permutations, rng
        )
        delta_rsa_cache[(a, b)] = res

        results["within_family_delta_geometry"][f"{a}_vs_{b}"] = {
            k: v for k, v in res.items()
            if k not in ("raw_a", "raw_b", "families", "pairs")
        }

        print(
            f"  {a:>6} vs {b:<6} | "
            f"r={res['pearson_r_family_zscored']:+.4f}  "
            f"rho={res['spearman_rho_family_zscored']:+.4f}  "
            f"perm-p={res['perm_p_r']:.6g}  z={res['z']:+.2f}"
        )

    print()
    print("2) BASELINE OFFSET REMOVED EXACTLY")
    print("   Center each state's 4 payload responses before computing geometry.")
    print("   Any shared '-baseline_LO(state)' term is mathematically gone.")
    print()

    centered_rs = []

    for a, b in task_pairs:
        res = state_label_perm_rsa_within_families(
            D_centered[a], D_centered[b],
            indices_by_family, args.permutations, rng
        )

        results["within_family_centered_geometry"][f"{a}_vs_{b}"] = {
            k: v for k, v in res.items()
            if k not in ("raw_a", "raw_b", "families", "pairs")
        }

        centered_rs.append(res["pearson_r_family_zscored"])

        print(
            f"  {a:>6} vs {b:<6} | "
            f"r={res['pearson_r_family_zscored']:+.4f}  "
            f"rho={res['spearman_rho_family_zscored']:+.4f}  "
            f"perm-p={res['perm_p_r']:.6g}  z={res['z']:+.2f}"
        )

    print()
    print("3) RAW PERTURBED LOG-ODDS GEOMETRY")
    print("   No baseline subtraction. Checks whether common structure exists directly")
    print("   in the perturbed outputs, not only in delta construction.")
    print()

    for a, b in task_pairs:
        res = state_label_perm_rsa_within_families(
            D_raw[a], D_raw[b],
            indices_by_family, args.permutations, rng
        )

        results["within_family_raw_perturbed_geometry"][f"{a}_vs_{b}"] = {
            k: v for k, v in res.items()
            if k not in ("raw_a", "raw_b", "families", "pairs")
        }

        print(
            f"  {a:>6} vs {b:<6} | "
            f"r={res['pearson_r_family_zscored']:+.4f}  "
            f"rho={res['spearman_rho_family_zscored']:+.4f}  "
            f"perm-p={res['perm_p_r']:.6g}"
        )

    print()
    print("4) PARTIAL CORRELATION CONTROLLING |BASELINE ΔLO|")
    print("   Same 140 within-family state pairs.")
    print()

    for a, b in task_pairs:
        res = delta_rsa_cache[(a, b)]
        va = res["raw_a"]
        vb = res["raw_b"]
        fams = res["families"]
        pairs_rows = res["pairs"]

        base_dist = np.array([
            abs(baseline[hashes[i]] - baseline[hashes[j]])
            for _, i, j in pairs_rows
        ], dtype=np.float64)

        # Family-zscore first, then partial out family-zscored baseline distance.
        za = family_zscore(va, fams)
        zb = family_zscore(vb, fams)
        zbase = family_zscore(base_dist, fams)

        pc = partial_corr(za, zb, zbase)

        # Permutation null within family at the pair-observation level.
        null = np.empty(args.permutations, dtype=np.float64)
        fam_arr = np.asarray(fams, dtype=object)
        unique = sorted(set(fam_arr.tolist()))

        for k in range(args.permutations):
            zbp = zb.copy()
            for fam in unique:
                ix = np.where(fam_arr == fam)[0]
                zbp[ix] = zb[rng.permutation(ix)]
            null[k] = partial_corr(za, zbp, zbase)

        p = (1 + np.sum(null >= pc)) / (args.permutations + 1)

        results["partial_corr_controlling_baseline_distance"][f"{a}_vs_{b}"] = {
            "partial_r": float(pc),
            "perm_p_one_sided": float(p),
        }

        print(
            f"  {a:>6} vs {b:<6} | partial r={pc:+.4f}  perm-p={p:.6g}"
        )

    print()
    print("5) ORIGINAL 20 MATCHED PAIRS, FAMILY EFFECT REMOVED")
    print("   Correlates pair separation after z-scoring the 4 pairs within each family.")
    print("   Cluster bootstrap resamples whole evidence families.")
    print()

    matched = []
    for pi, pair in enumerate(pairs, 1):
        ha = pair["a"]["order_hash"]
        hb = pair["b"]["order_hash"]
        ia = idx[ha]
        ib = idx[hb]
        row = {
            "pair": pi,
            "family": pair["a"]["family"],
        }
        for t in TASKS:
            row[t] = float(D_delta[t][ia, ib])
        matched.append(row)

    fams = np.array([r["family"] for r in matched], dtype=object)

    for a, b in task_pairs:
        xa = np.array([r[a] for r in matched])
        xb = np.array([r[b] for r in matched])

        res = bootstrap_matched_pairs_family_cluster(
            xa, xb, fams, args.bootstrap, rng
        )

        results["matched_pairs_family_deconfounded"][f"{a}_vs_{b}"] = res

        lo, hi = res["cluster_bootstrap_95_ci"]
        print(
            f"  {a:>6} vs {b:<6} | "
            f"within-family r={res['within_family_demeaned_r']:+.4f}  "
            f"cluster-bootstrap 95% CI=[{lo:+.4f}, {hi:+.4f}]"
        )

    print()
    print("=" * 96)
    print("GUARDRAIL")
    print("=" * 96)

    strong_centered = sum(r >= 0.40 for r in centered_rs)

    if strong_centered >= 2:
        verdict = "SURVIVES_BASELINE_AND_FAMILY_DECONFOUNDING"
        print(
            "Common structure survives after removing cross-family scale and "
            "exactly cancelling the shared baseline offset in at least two task-pairs."
        )
        print(
            "The baseline-subtraction / family-cluster confound is not sufficient "
            "to explain the earlier RSA result."
        )
    elif all(abs(r) < 0.20 for r in centered_rs):
        verdict = "EARLIER_RSA_LARGELY_EXPLAINED_BY_CONFOUND"
        print(
            "After exact baseline-offset removal and within-family control, "
            "cross-task geometry is weak."
        )
        print(
            "The earlier ~0.93-0.96 RSA was largely a shared-baseline/family artifact."
        )
    else:
        verdict = "MIXED_AFTER_DECONFOUNDING"
        print(
            "Some common structure remains, but the spectacular earlier RSA "
            "was partly inflated by shared baseline/family structure."
        )

    results["verdict"] = verdict

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print()
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
