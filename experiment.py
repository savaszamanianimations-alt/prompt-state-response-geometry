import argparse
import hashlib
import json
import math
import os
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "ibm-granite/granite-4.1-3b"
DEVICE = "cuda"

# Same synthetic branch as the existing controlled experiment.
VALUES = (
    "Igor Sysoev",   # target
    "Igor Sokolov",
    "Igor Smith",
    "Igor Petrov",
)
TARGET_VALUE = VALUES[0]
ANSWER_TEXT = " Igor Sysoev"  # leading space is intentional

# Families that were already useful in the earlier order-matched work.
# Each family is tested independently: within a family, evidence identity/counts
# are fixed and ONLY order is changed.
COUNT_FAMILIES = (
    (6, 1, 4, 1),
    (4, 3, 1, 2),
    (5, 0, 5, 4),
    (5, 4, 1, 2),
    (6, 5, 3, 0),
)

PAYLOADS = ("LUMA", "NOVA", "KITE", "4827")
TASKS = ("repeat", "copy", "write")

# IMPORTANT: freeze these literal perturbation strings for this validation.
# Pair selection happens BEFORE any perturbation scoring.
def companion_text(task, payload):
    if task == "repeat":
        return f"Repeat {payload}: ___"
    if task == "copy":
        return f"Copy {payload}: ___"
    if task == "write":
        return f"Write {payload}: ___"
    raise ValueError(task)

CONDITIONS = ["structured_only"] + [
    f"{task}_{payload.lower()}" for task in TASKS for payload in PAYLOADS
]


def condition_companion(condition):
    if condition == "structured_only":
        return None
    task, payload = condition.split("_", 1)
    return companion_text(task, payload.upper() if payload != "4827" else payload)


def family_key(counts):
    return "-".join(map(str, counts))


def order_hash(order):
    s = "\n".join(order).encode("utf-8")
    return hashlib.sha256(s).hexdigest()[:16]


def make_base_multiset(counts):
    assert len(counts) == len(VALUES)
    out = []
    for value, n in zip(VALUES, counts):
        out.extend([value] * int(n))
    return out


def unique_permutations_sample(counts, n, seed):
    """Sample unique random permutations without enumerating the full space."""
    base = make_base_multiset(counts)
    rng = random.Random(seed)
    seen = set()
    out = []

    # Number of distinct permutations, useful to cap impossible requests.
    total = math.factorial(len(base))
    for c in counts:
        total //= math.factorial(c)
    n = min(n, total)

    # Rejection sampling is fine here because n is far below total for our families.
    max_attempts = max(10000, n * 100)
    attempts = 0
    while len(out) < n and attempts < max_attempts:
        attempts += 1
        x = base.copy()
        rng.shuffle(x)
        t = tuple(x)
        if t not in seen:
            seen.add(t)
            out.append(t)

    if len(out) < n:
        raise RuntimeError(
            f"Could only sample {len(out)}/{n} unique permutations for {counts}."
        )
    return out, total


def make_task(order):
    # Deliberately NO record numbers. Every permutation has the exact same
    # literal line multiset; only line order changes.
    records = "\n".join(f"K7 -> {value}" for value in order)
    return (
        "Use only these temporary fictional records.\n"
        "Repeated observations may occur.\n\n"
        f"{records}\n\n"
        "Which value is best supported for K7?"
    )


def make_user_content(task_text, condition):
    companion = condition_companion(condition)
    if companion is None:
        return (
            "Complete the field independently.\n\n"
            f"{task_text} ___\n\n"
            "Output exactly:\n"
            "Target: <answer>"
        )
    return (
        "Complete both fields independently.\n\n"
        f"{task_text} ___\n"
        f"{companion}\n\n"
        "Output exactly:\n"
        "Target: <answer>\n"
        "Companion: <answer>"
    )


def prefix_ids_for(tokenizer, order, condition):
    task_text = make_task(order)
    content = make_user_content(task_text, condition)
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    ) + "Target:"
    return tokenizer.encode(rendered, add_special_tokens=False), rendered


def answer_boundary_ids(tokenizer):
    ids = tokenizer.encode(ANSWER_TEXT, add_special_tokens=False)
    toks = [tokenizer.decode([x]) for x in ids]
    if len(ids) < 2:
        raise RuntimeError(f"Unexpected answer tokenization: {toks}")
    return ids, toks


def append_jsonl(path, rows):
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_jsonl(path):
    rows = []
    if not Path(path).exists():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def entropy_and_metrics(logits, correct_id):
    # logits: [B, V], float32 from model. Metrics in float64 for stability.
    z = logits.double()
    correct = z[:, correct_id]

    others_z = z.clone()
    others_z[:, correct_id] = -torch.inf
    others = torch.logsumexp(others_z, dim=-1)
    logodds = correct - others

    logp = F.log_softmax(z, dim=-1)
    probs = logp.exp()
    entropy = -(probs * logp).sum(dim=-1)
    p_correct = probs[:, correct_id]
    rank = (z > correct.unsqueeze(1)).sum(dim=-1) + 1
    top1 = torch.argmax(z, dim=-1)

    return p_correct, logodds, entropy, rank, top1


@torch.inference_mode()
def score_items(model, tokenizer, items, condition, batch_size, answer_ids, correct_id):
    """Score a list of dicts containing family/order/hash under one condition."""
    results = []
    force_first = answer_ids[0]

    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        seqs = []
        raw_lengths = []
        for item in batch:
            prefix_ids, _ = prefix_ids_for(tokenizer, item["order"], condition)
            seq = prefix_ids + [force_first]
            seqs.append(seq)
            raw_lengths.append(len(seq))

        max_len = max(raw_lengths)
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        input_ids = torch.full(
            (len(seqs), max_len), pad_id, dtype=torch.long, device=DEVICE
        )
        attention_mask = torch.zeros(
            (len(seqs), max_len), dtype=torch.long, device=DEVICE
        )
        for i, seq in enumerate(seqs):
            L = len(seq)
            input_ids[i, :L] = torch.tensor(seq, dtype=torch.long, device=DEVICE)
            attention_mask[i, :L] = 1

        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        last_pos = attention_mask.sum(dim=1) - 1
        logits = out.logits[torch.arange(len(batch), device=DEVICE), last_pos].float()

        p, lo, h, rank, top1 = entropy_and_metrics(logits, correct_id)

        p = p.cpu().tolist()
        lo = lo.cpu().tolist()
        h = h.cpu().tolist()
        rank = rank.cpu().tolist()
        top1 = top1.cpu().tolist()

        for j, item in enumerate(batch):
            results.append({
                "family": item["family"],
                "counts": item["counts"],
                "order_hash": item["order_hash"],
                "order": item["order"],
                "condition": condition,
                "input_tokens": raw_lengths[j],
                "p_correct": float(p[j]),
                "p_correct_pct": float(p[j] * 100.0),
                "logodds": float(lo[j]),
                "entropy": float(h[j]),
                "rank": int(rank[j]),
                "top1_id": int(top1[j]),
                "top1_text": tokenizer.decode([int(top1[j])]),
                "correct_id": int(correct_id),
                "correct_text": tokenizer.decode([int(correct_id)]),
            })

        done = min(start + batch_size, len(items))
        print(f"    {condition}: {done}/{len(items)}", flush=True)

    return results


def verify_family_structure(tokenizer, family_items):
    """Strict pre-score audit: same evidence multiset, chars, baseline token count."""
    expected_counter = Counter(make_base_multiset(family_items[0]["counts"]))
    task_char_lengths = set()
    token_lengths = set()

    for item in family_items:
        if Counter(item["order"]) != expected_counter:
            raise AssertionError(f"Evidence multiset mismatch: {item['order_hash']}")
        task = make_task(item["order"])
        task_char_lengths.add(len(task))
        ids, _ = prefix_ids_for(tokenizer, item["order"], "structured_only")
        token_lengths.add(len(ids))

    return {
        "same_evidence_multiset": True,
        "task_char_lengths": sorted(task_char_lengths),
        "baseline_prefix_token_lengths": sorted(token_lengths),
        "same_task_char_length": len(task_char_lengths) == 1,
        "same_baseline_token_length": len(token_lengths) == 1,
    }


def baseline_pair_candidates(rows, max_dlo, max_dh, p_min, p_max):
    eligible = [
        r for r in rows
        if r["rank"] == 1 and p_min <= r["p_correct"] <= p_max
    ]
    eligible.sort(key=lambda r: r["logodds"])
    candidates = []

    # Sorted-window search: only compare nearby LO values.
    for i, a in enumerate(eligible):
        j = i + 1
        while j < len(eligible):
            b = eligible[j]
            dlo = abs(b["logodds"] - a["logodds"])
            if dlo > max_dlo:
                break
            dh = abs(b["entropy"] - a["entropy"])
            if dh <= max_dh:
                dp = abs(b["p_correct"] - a["p_correct"])
                # Selection score uses ONLY baseline-local metrics.
                score = dlo + 0.25 * dh + 0.25 * dp
                candidates.append({
                    "a": a,
                    "b": b,
                    "delta_logodds": dlo,
                    "delta_entropy": dh,
                    "delta_p": dp,
                    "delta_p_pp": dp * 100.0,
                    "selection_score": score,
                })
            j += 1

    candidates.sort(key=lambda x: (
        x["selection_score"], x["delta_logodds"], x["delta_entropy"]
    ))
    return candidates


def choose_disjoint(candidates, max_pairs):
    used = set()
    chosen = []
    for c in candidates:
        ha = c["a"]["order_hash"]
        hb = c["b"]["order_hash"]
        if ha in used or hb in used:
            continue
        chosen.append(c)
        used.add(ha)
        used.add(hb)
        if len(chosen) >= max_pairs:
            break
    return chosen


def pearson(xs, ys):
    if len(xs) < 2:
        return float("nan")
    mx = statistics.fmean(xs)
    my = statistics.fmean(ys)
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    den = math.sqrt(sum(x*x for x in dx) * sum(y*y for y in dy))
    if den == 0:
        return float("nan")
    return sum(x*y for x, y in zip(dx, dy)) / den


def summarize_pair(pair, profile_by_key):
    a0 = pair["a"]
    b0 = pair["b"]
    ha = a0["order_hash"]
    hb = b0["order_hash"]

    conds = [c for c in CONDITIONS if c != "structured_only"]
    avec, bvec = [], []
    rows = []
    flips_a = flips_b = 0

    for cond in conds:
        a = profile_by_key[(ha, cond)]
        b = profile_by_key[(hb, cond)]
        da = a["logodds"] - a0["logodds"]
        db = b["logodds"] - b0["logodds"]
        avec.append(da)
        bvec.append(db)
        flips_a += int(a["rank"] != 1)
        flips_b += int(b["rank"] != 1)
        rows.append({
            "condition": cond,
            "a_dlo": da,
            "b_dlo": db,
            "abs_profile_diff": abs(da - db),
            "a_p_pct": a["p_correct_pct"],
            "b_p_pct": b["p_correct_pct"],
            "a_rank": a["rank"],
            "b_rank": b["rank"],
        })

    diffs = [abs(a-b) for a, b in zip(avec, bvec)]
    sq = [(a-b)**2 for a, b in zip(avec, bvec)]

    return {
        "family": a0["family"],
        "counts": a0["counts"],
        "a_hash": ha,
        "b_hash": hb,
        "a_order": a0["order"],
        "b_order": b0["order"],
        "baseline_delta_logodds": pair["delta_logodds"],
        "baseline_delta_entropy": pair["delta_entropy"],
        "baseline_delta_p_pp": pair["delta_p_pp"],
        "a_baseline_p_pct": a0["p_correct_pct"],
        "b_baseline_p_pct": b0["p_correct_pct"],
        "a_baseline_logodds": a0["logodds"],
        "b_baseline_logodds": b0["logodds"],
        "profile_mae": statistics.fmean(diffs),
        "profile_rmse": math.sqrt(statistics.fmean(sq)),
        "profile_pearson": pearson(avec, bvec),
        "a_mean_dlo": statistics.fmean(avec),
        "b_mean_dlo": statistics.fmean(bvec),
        "abs_delta_mean_dlo": abs(statistics.fmean(avec) - statistics.fmean(bvec)),
        "a_mean_abs_dlo": statistics.fmean(abs(x) for x in avec),
        "b_mean_abs_dlo": statistics.fmean(abs(x) for x in bvec),
        "flip_count_a": flips_a,
        "flip_count_b": flips_b,
        "flip_count_difference": abs(flips_a - flips_b),
        "conditions": rows,
    }


def pair_condition_token_audit(tokenizer, pair):
    out = {}
    for cond in CONDITIONS:
        a_ids, _ = prefix_ids_for(tokenizer, pair["a"]["order"], cond)
        b_ids, _ = prefix_ids_for(tokenizer, pair["b"]["order"], cond)
        out[cond] = {
            "a": len(a_ids),
            "b": len(b_ids),
            "same": len(a_ids) == len(b_ids),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-family", type=int, default=768)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20260819)
    ap.add_argument("--max-dlo", type=float, default=0.01)
    ap.add_argument("--max-dh", type=float, default=0.02)
    ap.add_argument("--p-min", type=float, default=0.80)
    ap.add_argument("--p-max", type=float, default=0.995)
    ap.add_argument("--pairs-per-family", type=int, default=4)
    ap.add_argument("--outdir", default="random_order_validation")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    baseline_path = outdir / "baselines.jsonl"
    profile_path = outdir / "profiles.jsonl"
    pairs_path = outdir / "selected_pairs.json"
    final_path = outdir / "final_summary.json"

    print("Loading tokenizer/model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        dtype=torch.float32,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    model.to(DEVICE)
    model.eval()
    model.config.use_cache = False

    answer_ids, answer_tokens = answer_boundary_ids(tokenizer)
    correct_id = answer_ids[1]
    print("ANSWER TOKEN PATH:", answer_tokens)
    print("FORCE:", repr(answer_tokens[0]))
    print("SCORE:", repr(tokenizer.decode([correct_id])))
    print()

    # -------------------------
    # 1) Generate random states
    # -------------------------
    all_items = []
    structural_audits = {}
    for fi, counts in enumerate(COUNT_FAMILIES):
        fam = family_key(counts)
        perms, total = unique_permutations_sample(
            counts,
            args.n_per_family,
            args.seed + fi * 100003,
        )
        items = [{
            "family": fam,
            "counts": list(counts),
            "order": list(order),
            "order_hash": order_hash(order),
        } for order in perms]

        audit = verify_family_structure(tokenizer, items)
        structural_audits[fam] = audit
        print(f"FAMILY {fam}: sampled {len(items)} / {total} unique permutations")
        print("  structural audit:", audit)
        if not audit["same_task_char_length"]:
            raise AssertionError(f"Character-length mismatch in family {fam}")
        # Token count equality is expected; if it fails, we keep the family but
        # pairs will be audited condition-by-condition and mismatched pairs rejected.
        all_items.extend(items)

    # -------------------------
    # 2) Baseline scoring/resume
    # -------------------------
    old_base = load_jsonl(baseline_path)
    base_map = {(r["family"], r["order_hash"]): r for r in old_base}
    todo = [x for x in all_items if (x["family"], x["order_hash"]) not in base_map]
    print(f"\nBaseline checkpoint: {len(base_map)} existing, {len(todo)} to score")

    # Score family-by-family so every batch has identical evidence length.
    todo_by_family = defaultdict(list)
    for item in todo:
        todo_by_family[item["family"]].append(item)

    for fam, items in todo_by_family.items():
        print(f"\nBASELINE family {fam}")
        rows = score_items(
            model, tokenizer, items, "structured_only",
            args.batch_size, answer_ids, correct_id,
        )
        append_jsonl(baseline_path, rows)
        for r in rows:
            base_map[(r["family"], r["order_hash"])] = r

    # -------------------------
    # 3) Blind baseline pair selection
    # -------------------------
    selected = []
    matching_landscape = {}
    rows_by_family = defaultdict(list)
    for r in base_map.values():
        rows_by_family[r["family"]].append(r)

    print("\n" + "=" * 96)
    print("BASELINE MATCHING LANDSCAPE")
    print("Pair selection uses baseline metrics ONLY; no perturbation result is visible yet.")
    print("=" * 96)

    for fam in sorted(rows_by_family):
        rows = rows_by_family[fam]
        landscape = {}
        for t in (0.0025, 0.005, 0.01, 0.02, 0.05):
            cs = baseline_pair_candidates(rows, t, args.max_dh, args.p_min, args.p_max)
            landscape[str(t)] = len(cs)
        matching_landscape[fam] = landscape

        candidates = baseline_pair_candidates(
            rows, args.max_dlo, args.max_dh, args.p_min, args.p_max
        )
        chosen = choose_disjoint(candidates, args.pairs_per_family)

        # Strict condition-by-condition token-count audit. Reject any pair whose
        # rendered prompt lengths differ in even one condition.
        clean = []
        for c in chosen:
            audit = pair_condition_token_audit(tokenizer, c)
            if all(v["same"] for v in audit.values()):
                c = dict(c)
                c["token_audit"] = audit
                clean.append(c)
        selected.extend(clean)

        print(f"\nFAMILY {fam}")
        print("  candidate counts by ΔLO:", landscape)
        print(f"  selected clean disjoint pairs @ ΔLO<={args.max_dlo}: {len(clean)}")
        for i, c in enumerate(clean, 1):
            print(
                f"    {i}. {c['a']['order_hash']} vs {c['b']['order_hash']} | "
                f"ΔLO={c['delta_logodds']:.8f} "
                f"ΔP={c['delta_p_pp']:.6f}pp "
                f"ΔH={c['delta_entropy']:.8f}"
            )

    # Save pair selection BEFORE perturbation scoring.
    with open(pairs_path, "w", encoding="utf-8") as f:
        json.dump({
            "selection_rule": {
                "rank": 1,
                "p_min": args.p_min,
                "p_max": args.p_max,
                "max_delta_logodds": args.max_dlo,
                "max_delta_entropy": args.max_dh,
                "disjoint_pairs": True,
                "selection_uses_perturbation_results": False,
            },
            "matching_landscape": matching_landscape,
            "pairs": selected,
        }, f, indent=2, ensure_ascii=False)

    if not selected:
        print("\nNO PAIRS met the requested ultra-tight threshold.")
        print("Rerun with e.g. --n-per-family 1500, or inspect landscape before relaxing ΔLO.")
        print("Nothing was perturbation-scored, so there is no outcome-based selection bias.")
        return

    # -------------------------
    # 4) Perturbation profiles ONLY after selection is frozen
    # -------------------------
    selected_items = {}
    for pair in selected:
        for side in ("a", "b"):
            r = pair[side]
            selected_items[(r["family"], r["order_hash"])] = {
                "family": r["family"],
                "counts": r["counts"],
                "order": r["order"],
                "order_hash": r["order_hash"],
            }

    old_prof = load_jsonl(profile_path)
    prof_map = {(r["order_hash"], r["condition"]): r for r in old_prof}

    # Baseline is already known; profile file stores nonbaseline cells.
    selected_by_family = defaultdict(list)
    for item in selected_items.values():
        selected_by_family[item["family"]].append(item)

    for cond in CONDITIONS[1:]:
        print(f"\nPROFILE CONDITION {cond}")
        for fam, fam_items in selected_by_family.items():
            todo_cond = [
                item for item in fam_items
                if (item["order_hash"], cond) not in prof_map
            ]
            if not todo_cond:
                continue
            rows = score_items(
                model, tokenizer, todo_cond, cond,
                args.batch_size, answer_ids, correct_id,
            )
            append_jsonl(profile_path, rows)
            for r in rows:
                prof_map[(r["order_hash"], r["condition"])] = r

    # -------------------------
    # 5) Summaries
    # -------------------------
    summaries = [summarize_pair(pair, prof_map) for pair in selected]
    summaries.sort(key=lambda x: x["baseline_delta_logodds"])

    print("\n" + "=" * 96)
    print("FINAL RANDOM-ORDER VALIDATION")
    print("=" * 96)
    for i, s in enumerate(summaries, 1):
        print(
            f"{i:02d}. family={s['family']}  "
            f"ΔLO={s['baseline_delta_logodds']:.8f}  "
            f"ΔP={s['baseline_delta_p_pp']:.6f}pp  "
            f"profile_MAE={s['profile_mae']:.4f}  "
            f"RMSE={s['profile_rmse']:.4f}  "
            f"r={s['profile_pearson']:+.4f}  "
            f"flips={s['flip_count_a']}/{s['flip_count_b']}"
        )

    maes = [s["profile_mae"] for s in summaries]
    dlos = [s["baseline_delta_logodds"] for s in summaries]
    print("\nAGGREGATE")
    print("  N pairs:", len(summaries))
    print(f"  median baseline ΔLO: {statistics.median(dlos):.8f}")
    print(f"  mean profile MAE:    {statistics.fmean(maes):.4f}")
    print(f"  median profile MAE:  {statistics.median(maes):.4f}")
    print(f"  max profile MAE:     {max(maes):.4f}")
    if len(summaries) >= 2:
        print(f"  corr(ΔLO, MAE):      {pearson(dlos, maes):+.4f}")

    final = {
        "model": MODEL,
        "device": DEVICE,
        "dtype": "float32",
        "attn_implementation": "eager",
        "use_cache": False,
        "seed": args.seed,
        "n_per_family": args.n_per_family,
        "count_families": [list(x) for x in COUNT_FAMILIES],
        "values": list(VALUES),
        "answer_token_path": answer_tokens,
        "forced_token": answer_tokens[0],
        "scored_token": tokenizer.decode([correct_id]),
        "conditions": CONDITIONS,
        "structural_audits": structural_audits,
        "matching_landscape": matching_landscape,
        "selection_rule": {
            "rank": 1,
            "p_min": args.p_min,
            "p_max": args.p_max,
            "max_delta_logodds": args.max_dlo,
            "max_delta_entropy": args.max_dh,
            "pairs_per_family": args.pairs_per_family,
            "disjoint": True,
            "outcome_blind": True,
        },
        "pair_summaries": summaries,
        "aggregate": {
            "n_pairs": len(summaries),
            "median_baseline_delta_logodds": statistics.median(dlos),
            "mean_profile_mae": statistics.fmean(maes),
            "median_profile_mae": statistics.median(maes),
            "max_profile_mae": max(maes),
            "pearson_baseline_delta_lo_vs_mae": pearson(dlos, maes) if len(summaries) >= 2 else None,
        },
    }
    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)

    print(f"\nSaved:\n  {baseline_path}\n  {pairs_path}\n  {profile_path}\n  {final_path}")


if __name__ == "__main__":
    main()
