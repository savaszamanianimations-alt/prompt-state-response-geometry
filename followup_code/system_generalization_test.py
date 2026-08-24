import argparse
import json
import math
import os
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# CROSS-INTERVENTION GENERALIZATION TEST
#
# Question:
#   Can geometry built ONLY from the old prompt perturbations
#   (repeat/copy/write) predict geometry under an unseen
#   intervention channel: SYSTEM PROMPTS?
#
# Requires:
#   full_distribution_validation/selected_pairs.json
#   full_distribution_validation/profiles.jsonl
#
# Does fresh model inference ONLY for:
#   - exact no-system baseline recheck
#   - 8 unseen system-prompt interventions
#
# No system-prompt result is used to construct the old geometry.
# ============================================================

MODEL = "ibm-granite/granite-4.1-3b"
DEVICE = "cuda"

ANSWER_TEXT = " Igor Sysoev"

OLD_TASKS = ("repeat", "copy", "write")
PAYLOADS = ("luma", "nova", "kite", "4827")
OLD_CONDS = {
    task: [f"{task}_{p}" for p in PAYLOADS]
    for task in OLD_TASKS
}

# These were NOT present in the original experiment.
# They operate through the system-message channel, not by adding a
# companion field to the user prompt.
SYSTEM_PROMPTS = [
    "Follow the user's requested output format exactly.",
    "Answer concisely and literally. Do not add explanations.",
    "Use only the evidence provided in the user's message.",
    "Treat this as a deterministic text-completion task and be precise.",
    "Prioritize the explicit records in the user message over outside knowledge.",
    "Return only the requested fields, with no commentary.",
    "Read the evidence carefully and complete the task exactly as written.",
    "Be consistent, precise, and faithful to the user's stated format.",
]


# ----------------------------
# Original task reconstruction
# ----------------------------

def make_task(order):
    records = "\n".join(f"K7 -> {value}" for value in order)
    return (
        "Use only these temporary fictional records.\n"
        "Repeated observations may occur.\n\n"
        f"{records}\n\n"
        "Which value is best supported for K7?"
    )


def make_user_content_no_companion(task_text):
    return (
        "Complete the field independently.\n\n"
        f"{task_text} ___\n\n"
        "Output exactly:\n"
        "Target: <answer>"
    )


def prefix_ids_for_system(tokenizer, order, system_prompt):
    task_text = make_task(order)
    content = make_user_content_no_companion(task_text)

    messages = []
    if system_prompt is not None:
        messages.append({
            "role": "system",
            "content": system_prompt,
        })
    messages.append({
        "role": "user",
        "content": content,
    })

    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    ) + "Target:"

    ids = tokenizer.encode(
        rendered,
        add_special_tokens=False,
    )
    return ids, rendered


def answer_boundary_ids(tokenizer):
    ids = tokenizer.encode(
        ANSWER_TEXT,
        add_special_tokens=False,
    )
    toks = [tokenizer.decode([x]) for x in ids]

    if len(ids) < 2:
        raise RuntimeError(
            f"Unexpected answer tokenization: {toks}"
        )

    return ids, toks


# ----------------------------
# IO helpers
# ----------------------------

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path, rows):
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(row, ensure_ascii=False)
                + "\n"
            )
        f.flush()
        os.fsync(f.fileno())


# ----------------------------
# Metrics
# ----------------------------

def target_metrics(logits, correct_id):
    z = logits.double()

    correct = z[:, correct_id]

    others_z = z.clone()
    others_z[:, correct_id] = -torch.inf
    others = torch.logsumexp(
        others_z,
        dim=-1,
    )

    logodds = correct - others

    logp = F.log_softmax(z, dim=-1)
    p = logp[:, correct_id].exp()

    return p, logodds


@torch.inference_mode()
def score_system_condition(
    model,
    tokenizer,
    items,
    system_prompt,
    condition_name,
    batch_size,
    answer_ids,
    correct_id,
):
    force_first = answer_ids[0]
    rows = []

    for start in range(
        0,
        len(items),
        batch_size,
    ):
        batch = items[
            start:start + batch_size
        ]

        seqs = []
        raw_lengths = []

        for item in batch:
            prefix, _ = prefix_ids_for_system(
                tokenizer,
                item["order"],
                system_prompt,
            )

            seq = prefix + [force_first]
            seqs.append(seq)
            raw_lengths.append(len(seq))

        max_len = max(raw_lengths)

        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id

        input_ids = torch.full(
            (len(seqs), max_len),
            pad_id,
            dtype=torch.long,
            device=DEVICE,
        )

        attention_mask = torch.zeros(
            (len(seqs), max_len),
            dtype=torch.long,
            device=DEVICE,
        )

        for i, seq in enumerate(seqs):
            L = len(seq)
            input_ids[i, :L] = torch.tensor(
                seq,
                dtype=torch.long,
                device=DEVICE,
            )
            attention_mask[i, :L] = 1

        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )

        last_pos = (
            attention_mask.sum(dim=1) - 1
        )

        logits = out.logits[
            torch.arange(
                len(batch),
                device=DEVICE,
            ),
            last_pos,
        ].float()

        p, lo = target_metrics(
            logits,
            correct_id,
        )

        p = p.cpu().tolist()
        lo = lo.cpu().tolist()

        for i, item in enumerate(batch):
            rows.append({
                "family": item["family"],
                "order_hash": item["order_hash"],
                "condition": condition_name,
                "system_prompt": system_prompt,
                "input_tokens": raw_lengths[i],
                "p_correct": float(p[i]),
                "logodds": float(lo[i]),
            })

        done = min(
            start + batch_size,
            len(items),
        )
        print(
            f"    {condition_name}: "
            f"{done}/{len(items)}",
            flush=True,
        )

    return rows


def pearson(x, y):
    x = np.asarray(
        x,
        dtype=np.float64,
    )
    y = np.asarray(
        y,
        dtype=np.float64,
    )

    if len(x) < 2:
        return float("nan")

    x = x - x.mean()
    y = y - y.mean()

    den = math.sqrt(
        float(
            np.dot(x, x)
            * np.dot(y, y)
        )
    )

    if den == 0:
        return float("nan")

    return float(
        np.dot(x, y) / den
    )


def rankdata(x):
    x = np.asarray(x)
    order = np.argsort(
        x,
        kind="mergesort",
    )
    ranks = np.empty(
        len(x),
        dtype=np.float64,
    )

    i = 0
    while i < len(x):
        j = i + 1

        while (
            j < len(x)
            and x[order[j]]
            == x[order[i]]
        ):
            j += 1

        ranks[order[i:j]] = (
            i + j - 1
        ) / 2.0

        i = j

    return ranks


def spearman(x, y):
    return pearson(
        rankdata(x),
        rankdata(y),
    )


def mae_matrix(X):
    X = np.asarray(
        X,
        dtype=np.float64,
    )

    return np.abs(
        X[:, None, :]
        - X[None, :, :]
    ).mean(axis=2)


def family_zscore(
    values,
    families,
):
    values = np.asarray(
        values,
        dtype=np.float64,
    )
    families = np.asarray(
        families,
        dtype=object,
    )

    out = np.empty_like(values)

    for fam in sorted(
        set(families.tolist())
    ):
        mask = families == fam
        v = values[mask]

        sd = v.std(ddof=0)

        if sd > 0:
            out[mask] = (
                v - v.mean()
            ) / sd
        else:
            out[mask] = 0.0

    return out


def within_family_pairs(
    indices_by_family,
):
    rows = []

    for fam, idxs in sorted(
        indices_by_family.items()
    ):
        for ai in range(len(idxs)):
            for bi in range(
                ai + 1,
                len(idxs),
            ):
                rows.append(
                    (
                        fam,
                        idxs[ai],
                        idxs[bi],
                    )
                )

    return rows


def rsa_within_family_permutation(
    D_seen,
    D_unseen,
    indices_by_family,
    n_perm,
    rng,
):
    """
    Strict cross-intervention RSA:
    - ONLY within-family state pairs
    - family distance scale removed by z-scoring
    - unseen state labels permuted independently inside each evidence family
    """
    pairs = within_family_pairs(
        indices_by_family
    )

    seen = np.array([
        D_seen[i, j]
        for _, i, j in pairs
    ], dtype=np.float64)

    unseen = np.array([
        D_unseen[i, j]
        for _, i, j in pairs
    ], dtype=np.float64)

    fams = np.array([
        fam
        for fam, _, _ in pairs
    ], dtype=object)

    seen_z = family_zscore(
        seen,
        fams,
    )
    unseen_z = family_zscore(
        unseen,
        fams,
    )

    obs_r = pearson(
        seen_z,
        unseen_z,
    )
    obs_rho = spearman(
        seen_z,
        unseen_z,
    )

    null_r = np.empty(
        n_perm,
        dtype=np.float64,
    )
    null_rho = np.empty(
        n_perm,
        dtype=np.float64,
    )

    n_states = D_seen.shape[0]

    fam_idx = {
        fam: np.asarray(
            idxs,
            dtype=int,
        )
        for fam, idxs
        in indices_by_family.items()
    }

    for k in range(n_perm):
        mapping = np.arange(n_states)

        for fam, idxs in fam_idx.items():
            mapping[idxs] = (
                rng.permutation(idxs)
            )

        unseen_perm = np.array([
            D_unseen[
                mapping[i],
                mapping[j],
            ]
            for _, i, j in pairs
        ], dtype=np.float64)

        unseen_perm_z = family_zscore(
            unseen_perm,
            fams,
        )

        null_r[k] = pearson(
            seen_z,
            unseen_perm_z,
        )

        null_rho[k] = spearman(
            seen_z,
            unseen_perm_z,
        )

    p_r = (
        1 + np.sum(null_r >= obs_r)
    ) / (n_perm + 1)

    p_rho = (
        1
        + np.sum(
            null_rho >= obs_rho
        )
    ) / (n_perm + 1)

    sd = null_r.std(ddof=1)

    z = (
        (obs_r - null_r.mean()) / sd
        if sd > 0
        else float("inf")
    )

    return {
        "n_within_family_distances":
            len(pairs),
        "pearson_r": float(obs_r),
        "spearman_rho":
            float(obs_rho),
        "perm_p_r_one_sided":
            float(p_r),
        "perm_p_rho_one_sided":
            float(p_rho),
        "null_mean":
            float(null_r.mean()),
        "null_sd":
            float(sd),
        "z_vs_null":
            float(z),
    }


def nearest_neighbor_transfer(
    D_seen,
    D_unseen,
):
    """
    Old geometry chooses the nearest state.
    How highly is that exact state ranked in unseen system geometry?
    """
    n = D_seen.shape[0]

    ranks = []
    percentiles = []

    for i in range(n):
        candidates = [
            j
            for j in range(n)
            if j != i
        ]

        nearest = min(
            candidates,
            key=lambda j:
                D_seen[i, j],
        )

        target_order = sorted(
            candidates,
            key=lambda j:
                D_unseen[i, j],
        )

        rank = (
            target_order.index(nearest)
            + 1
        )

        ranks.append(rank)

        pct = (
            (rank - 1) / (n - 2)
            if n > 2
            else 0.0
        )
        percentiles.append(pct)

    return {
        "median_rank":
            float(
                statistics.median(
                    ranks
                )
            ),
        "mean_rank":
            float(
                statistics.fmean(
                    ranks
                )
            ),
        "median_normalized_percentile":
            float(
                statistics.median(
                    percentiles
                )
            ),
        "fraction_top5":
            float(
                sum(r <= 5 for r in ranks)
                / len(ranks)
            ),
        "fraction_top10":
            float(
                sum(r <= 10 for r in ranks)
                / len(ranks)
            ),
    }


def susceptibility(
    X,
):
    X = np.asarray(
        X,
        dtype=np.float64,
    )

    return np.abs(X).mean(axis=1)


def susceptibility_perm_test(
    seen_susc,
    unseen_susc,
    family_by_state,
    n_perm,
    rng,
):
    fams = np.asarray(
        family_by_state,
        dtype=object,
    )

    a = family_zscore(
        seen_susc,
        fams,
    )
    b = family_zscore(
        unseen_susc,
        fams,
    )

    obs = pearson(a, b)
    obs_s = spearman(a, b)

    null = np.empty(
        n_perm,
        dtype=np.float64,
    )

    unique = sorted(
        set(fams.tolist())
    )

    for k in range(n_perm):
        bp = b.copy()

        for fam in unique:
            idx = np.where(
                fams == fam
            )[0]

            bp[idx] = b[
                rng.permutation(idx)
            ]

        null[k] = pearson(
            a,
            bp,
        )

    p = (
        1 + np.sum(null >= obs)
    ) / (n_perm + 1)

    return {
        "pearson_r_family_zscored":
            float(obs),
        "spearman_rho_family_zscored":
            float(obs_s),
        "perm_p_r_one_sided":
            float(p),
    }


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--pairs",
        default=(
            "full_distribution_validation/"
            "selected_pairs.json"
        ),
    )

    ap.add_argument(
        "--profiles",
        default=(
            "full_distribution_validation/"
            "profiles.jsonl"
        ),
    )

    ap.add_argument(
        "--system-cache",
        default=(
            "system_generalization_profiles.jsonl"
        ),
    )

    ap.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    ap.add_argument(
        "--permutations",
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
        default=(
            "system_generalization_summary.json"
        ),
    )

    args = ap.parse_args()

    pairs_path = Path(args.pairs)
    profiles_path = Path(args.profiles)

    if not pairs_path.exists():
        raise FileNotFoundError(
            f"Missing {pairs_path}"
        )

    if not profiles_path.exists():
        raise FileNotFoundError(
            f"Missing {profiles_path}"
        )

    with open(
        pairs_path,
        "r",
        encoding="utf-8",
    ) as f:
        pair_data = json.load(f)

    pairs = pair_data["pairs"]

    old_profiles = load_jsonl(
        profiles_path
    )

    # ----------------------------------------
    # Recover exact 40 states + old baseline.
    # ----------------------------------------

    items = {}
    saved_baseline = {}
    family = {}

    for pair in pairs:
        for side in ("a", "b"):
            r = pair[side]
            h = r["order_hash"]

            items[h] = {
                "family": r["family"],
                "order_hash": h,
                "order": r["order"],
            }

            saved_baseline[h] = float(
                r["logodds"]
            )
            family[h] = r["family"]

    hashes = sorted(items)
    item_list = [
        items[h]
        for h in hashes
    ]

    idx = {
        h: i
        for i, h in enumerate(hashes)
    }

    n = len(hashes)

    indices_by_family = defaultdict(list)

    for h in hashes:
        indices_by_family[
            family[h]
        ].append(idx[h])

    family_by_state = [
        family[h]
        for h in hashes
    ]

    # ----------------------------------------
    # Build OLD geometry from old outcomes only.
    # This is frozen before new inference.
    # ----------------------------------------

    old_lo = {}

    for r in old_profiles:
        h = r["order_hash"]
        cond = r["condition"]

        if h in idx:
            old_lo[
                (h, cond)
            ] = float(r["logodds"])

    X_seen_parts = []

    for task in OLD_TASKS:
        X = np.zeros(
            (n, 4),
            dtype=np.float64,
        )

        for i, h in enumerate(hashes):
            for j, cond in enumerate(
                OLD_CONDS[task]
            ):
                key = (h, cond)

                if key not in old_lo:
                    raise RuntimeError(
                        f"Missing old profile: "
                        f"{h} {cond}"
                    )

                X[i, j] = (
                    old_lo[key]
                    - saved_baseline[h]
                )

        # IMPORTANT:
        # Remove the common per-state baseline offset
        # exactly, as in the deconfounded test.
        X = X - X.mean(
            axis=1,
            keepdims=True,
        )

        X_seen_parts.append(X)

    # 40 x 12 fingerprint, constructed before any
    # system-prompt outcome is scored/read.
    X_seen = np.concatenate(
        X_seen_parts,
        axis=1,
    )

    D_seen = mae_matrix(X_seen)

    seen_susc = susceptibility(
        X_seen
    )

    print("=" * 96)
    print(
        "CROSS-INTERVENTION GENERALIZATION:"
        " OLD PROMPT GEOMETRY -> UNSEEN SYSTEM PROMPTS"
    )
    print("=" * 96)
    print(f"States: {n}")
    print(
        "Old geometry: 12 coordinates "
        "(repeat/copy/write), family-centered."
    )
    print(
        "New intervention channel: "
        f"{len(SYSTEM_PROMPTS)} system prompts."
    )
    print(
        "No system-prompt outcome has been used "
        "to construct D_seen."
    )
    print()

    # ----------------------------------------
    # Load model + fresh unseen inference.
    # ----------------------------------------

    print("Loading model...", flush=True)

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL
        )
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    tokenizer.padding_side = "right"

    # Preflight system-role support.
    try:
        _, preview = prefix_ids_for_system(
            tokenizer,
            item_list[0]["order"],
            SYSTEM_PROMPTS[0],
        )
    except Exception as e:
        raise RuntimeError(
            "The tokenizer/chat template rejected "
            "a system-role message.\n"
            f"Original error: {e}"
        )

    print(
        "System-role template preflight: OK"
    )

    model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL,
            dtype=torch.float32,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
        )
    )

    model.to(DEVICE)
    model.eval()
    model.config.use_cache = False

    answer_ids, answer_tokens = (
        answer_boundary_ids(
            tokenizer
        )
    )
    correct_id = answer_ids[1]

    print(
        "ANSWER TOKEN PATH:",
        answer_tokens,
    )
    print()

    # ----------------------------------------
    # Cache makes the test restartable.
    # ----------------------------------------

    cache_path = Path(
        args.system_cache
    )

    cached_rows = load_jsonl(
        cache_path
    ) if cache_path.exists() else []

    cache = {
        (
            r["order_hash"],
            r["condition"],
        ): r
        for r in cached_rows
    }

    conditions = [
        (
            "no_system_recheck",
            None,
        )
    ] + [
        (
            f"system_{i:02d}",
            prompt,
        )
        for i, prompt
        in enumerate(
            SYSTEM_PROMPTS,
            1,
        )
    ]

    for condition_name, system_prompt in conditions:
        todo = [
            item
            for item in item_list
            if (
                item["order_hash"],
                condition_name,
            ) not in cache
        ]

        if not todo:
            print(
                f"{condition_name}: cached"
            )
            continue

        print(
            f"\nSCORING {condition_name}"
        )

        rows = score_system_condition(
            model,
            tokenizer,
            todo,
            system_prompt,
            condition_name,
            args.batch_size,
            answer_ids,
            correct_id,
        )

        append_jsonl(
            cache_path,
            rows,
        )

        for r in rows:
            cache[
                (
                    r["order_hash"],
                    r["condition"],
                )
            ] = r

    # ----------------------------------------
    # Deterministic baseline sanity check.
    # ----------------------------------------

    baseline_diffs = []

    for h in hashes:
        fresh = cache[
            (
                h,
                "no_system_recheck",
            )
        ]["logodds"]

        baseline_diffs.append(
            abs(
                fresh
                - saved_baseline[h]
            )
        )

    print()
    print(
        "BASELINE RECHECK"
    )
    print(
        "  median |fresh - saved| ΔLO:",
        f"{statistics.median(baseline_diffs):.12g}"
    )
    print(
        "  max    |fresh - saved| ΔLO:",
        f"{max(baseline_diffs):.12g}"
    )

    # Use fresh no-system baseline for new deltas.
    fresh_baseline = {
        h: float(
            cache[
                (
                    h,
                    "no_system_recheck",
                )
            ]["logodds"]
        )
        for h in hashes
    }

    # ----------------------------------------
    # Build unseen system-prompt response space.
    # ----------------------------------------

    X_system = np.zeros(
        (
            n,
            len(SYSTEM_PROMPTS),
        ),
        dtype=np.float64,
    )

    for i, h in enumerate(hashes):
        base = fresh_baseline[h]

        for j in range(
            len(SYSTEM_PROMPTS)
        ):
            condition_name = (
                f"system_{j + 1:02d}"
            )

            lo = float(
                cache[
                    (
                        h,
                        condition_name,
                    )
                ]["logodds"]
            )

            X_system[i, j] = (
                lo - base
            )

    # Exact baseline-offset/common-system-shift removal:
    # compare the SHAPE across system prompts, not merely
    # whether a state is globally shifted by "having a system message".
    X_system_centered = (
        X_system
        - X_system.mean(
            axis=1,
            keepdims=True,
        )
    )

    D_system = mae_matrix(
        X_system_centered
    )

    system_susc = susceptibility(
        X_system_centered
    )

    # ----------------------------------------
    # Primary test: old geometry -> unseen geometry.
    # ----------------------------------------

    rng = np.random.default_rng(
        args.seed
    )

    rsa = rsa_within_family_permutation(
        D_seen,
        D_system,
        indices_by_family,
        args.permutations,
        rng,
    )

    print()
    print("=" * 96)
    print(
        "PRIMARY: OLD GEOMETRY -> "
        "UNSEEN SYSTEM GEOMETRY"
    )
    print("=" * 96)
    print(
        "  within-family distances:",
        rsa[
            "n_within_family_distances"
        ],
    )
    print(
        "  Pearson r:",
        f"{rsa['pearson_r']:+.4f}",
    )
    print(
        "  Spearman rho:",
        f"{rsa['spearman_rho']:+.4f}",
    )
    print(
        "  permutation p(r):",
        f"{rsa['perm_p_r_one_sided']:.6g}",
    )
    print(
        "  z vs permuted null:",
        f"{rsa['z_vs_null']:+.2f}",
    )

    # ----------------------------------------
    # Secondary: local neighbor transfer.
    # ----------------------------------------

    nn = nearest_neighbor_transfer(
        D_seen,
        D_system,
    )

    print()
    print(
        "LOCAL NEIGHBOR TRANSFER"
    )
    print(
        "  random median rank ~20/39"
    )
    print(
        "  median unseen-system rank:",
        f"{nn['median_rank']:.1f}/39",
    )
    print(
        "  median normalized percentile:",
        f"{nn['median_normalized_percentile']:.3f}",
    )
    print(
        "  fraction top-5:",
        f"{nn['fraction_top5']:.3f}",
    )

    # ----------------------------------------
    # Secondary: scalar susceptibility transfer.
    # ----------------------------------------

    susc = susceptibility_perm_test(
        seen_susc,
        system_susc,
        family_by_state,
        args.permutations,
        rng,
    )

    print()
    print(
        "PER-STATE SUSCEPTIBILITY TRANSFER"
    )
    print(
        "  family-zscored Pearson r:",
        f"{susc['pearson_r_family_zscored']:+.4f}",
    )
    print(
        "  family-zscored Spearman rho:",
        f"{susc['spearman_rho_family_zscored']:+.4f}",
    )
    print(
        "  permutation p(r):",
        f"{susc['perm_p_r_one_sided']:.6g}",
    )

    # ----------------------------------------
    # Matched-pair-only descriptive check.
    # ----------------------------------------

    pair_seen = []
    pair_system = []
    pair_fams = []

    for pair in pairs:
        ha = pair["a"]["order_hash"]
        hb = pair["b"]["order_hash"]

        ia = idx[ha]
        ib = idx[hb]

        pair_seen.append(
            D_seen[ia, ib]
        )
        pair_system.append(
            D_system[ia, ib]
        )
        pair_fams.append(
            pair["a"]["family"]
        )

    pair_seen_z = family_zscore(
        pair_seen,
        pair_fams,
    )
    pair_system_z = family_zscore(
        pair_system,
        pair_fams,
    )

    matched_r = pearson(
        pair_seen_z,
        pair_system_z,
    )

    print()
    print(
        "ORIGINAL 20 MATCHED PAIRS"
    )
    print(
        "  within-family-zscored old-vs-system distance r:",
        f"{matched_r:+.4f}",
    )

    # ----------------------------------------
    # Conservative guardrail.
    # ----------------------------------------

    print()
    print("=" * 96)
    print("GUARDRAIL")
    print("=" * 96)

    if (
        rsa["pearson_r"] >= 0.40
        and
        rsa["perm_p_r_one_sided"]
        <= 0.01
    ):
        verdict = (
            "CROSS_INTERVENTION_GENERALIZATION"
        )
        print(
            "Old prompt-response geometry "
            "predicts a substantial fraction of "
            "state-pair structure under unseen "
            "system-prompt interventions."
        )
        print(
            "This is evidence for shared "
            "state-dependent susceptibility across "
            "intervention channels."
        )
    elif (
        abs(rsa["pearson_r"])
        < 0.20
        and
        rsa["perm_p_r_one_sided"]
        > 0.05
    ):
        verdict = (
            "NO_CROSS_INTERVENTION_GENERALIZATION"
        )
        print(
            "The old prompt geometry does not "
            "generalize to the unseen system-prompt channel."
        )
        print(
            "That would substantially narrow the "
            "response-geometry interpretation."
        )
    else:
        verdict = "MIXED"
        print(
            "Some cross-intervention signal exists, "
            "but it is not clean enough for a strong claim."
        )

    summary = {
        "model": MODEL,
        "dtype": "float32",
        "attn_implementation": "eager",
        "use_cache": False,
        "n_states": n,
        "old_geometry_coordinates": 12,
        "system_prompts": SYSTEM_PROMPTS,
        "system_geometry_coordinates":
            len(SYSTEM_PROMPTS),
        "baseline_recheck": {
            "median_abs_logodds_diff":
                statistics.median(
                    baseline_diffs
                ),
            "max_abs_logodds_diff":
                max(baseline_diffs),
        },
        "primary_rsa":
            rsa,
        "nearest_neighbor_transfer":
            nn,
        "susceptibility_transfer":
            susc,
        "matched_pair_old_vs_system_r":
            float(matched_r),
        "verdict":
            verdict,
    }

    with open(
        args.out,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("Saved profiles:", cache_path)
    print("Saved summary:", args.out)


if __name__ == "__main__":
    main()
