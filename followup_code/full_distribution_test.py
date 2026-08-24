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


# ============================================================
# Standalone full-distribution validation
# No repo files are required.
# ============================================================

MODEL = "ibm-granite/granite-4.1-3b"
DEVICE = "cuda"

VALUES = (
    "Igor Sysoev",   # target
    "Igor Sokolov",
    "Igor Smith",
    "Igor Petrov",
)
TARGET_VALUE = VALUES[0]
ANSWER_TEXT = " Igor Sysoev"  # leading space is intentional

COUNT_FAMILIES = (
    (6, 1, 4, 1),
    (4, 3, 1, 2),
    (5, 0, 5, 4),
    (5, 4, 1, 2),
    (6, 5, 3, 0),
)

PAYLOADS = ("LUMA", "NOVA", "KITE", "4827")
TASKS = ("repeat", "copy", "write")

CONDITIONS = ["structured_only"] + [
    f"{task}_{payload.lower()}"
    for task in TASKS
    for payload in PAYLOADS
]


def companion_text(task, payload):
    if task == "repeat":
        return f"Repeat {payload}: ___"
    if task == "copy":
        return f"Copy {payload}: ___"
    if task == "write":
        return f"Write {payload}: ___"
    raise ValueError(task)


def condition_companion(condition):
    if condition == "structured_only":
        return None
    task, payload = condition.split("_", 1)
    payload = payload.upper() if payload != "4827" else payload
    return companion_text(task, payload)


def family_key(counts):
    return "-".join(map(str, counts))


def order_hash(order):
    s = "\n".join(order).encode("utf-8")
    return hashlib.sha256(s).hexdigest()[:16]


def make_base_multiset(counts):
    out = []
    for value, n in zip(VALUES, counts):
        out.extend([value] * int(n))
    return out


def unique_permutations_sample(counts, n, seed):
    base = make_base_multiset(counts)
    rng = random.Random(seed)

    total = math.factorial(len(base))
    for c in counts:
        total //= math.factorial(c)
    n = min(n, total)

    seen = set()
    out = []
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
            f"Could only sample {len(out)}/{n} unique permutations for {counts}"
        )

    return out, total


def make_task(order):
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
    path = Path(path)
    if not path.exists():
        return []

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def entropy_and_metrics(logits, correct_id):
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
def score_items(
    model,
    tokenizer,
    items,
    condition,
    batch_size,
    answer_ids,
    correct_id,
):
    results = []
    force_first = answer_ids[0]

    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]

        seqs = []
        raw_lengths = []

        for item in batch:
            prefix_ids, _ = prefix_ids_for(
                tokenizer, item["order"], condition
            )
            seq = prefix_ids + [force_first]
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
                seq, dtype=torch.long, device=DEVICE
            )
            attention_mask[i, :L] = 1

        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )

        last_pos = attention_mask.sum(dim=1) - 1
        logits = out.logits[
            torch.arange(len(batch), device=DEVICE),
            last_pos,
        ].float()

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
        print(
            f"    {condition}: {done}/{len(items)}",
            flush=True,
        )

    return results


@torch.inference_mode()
def score_full_logprobs(
    model,
    tokenizer,
    items,
    batch_size,
    answer_ids,
):
    """
    Re-score baseline states and keep COMPLETE next-token distributions.
    These are used ONLY for baseline pair matching.
    """
    result = {}
    force_first = answer_ids[0]

    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]

        seqs = []
        raw_lengths = []

        for item in batch:
            prefix_ids, _ = prefix_ids_for(
                tokenizer, item["order"], "structured_only"
            )
            seq = prefix_ids + [force_first]
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
                seq, dtype=torch.long, device=DEVICE
            )
            attention_mask[i, :L] = 1

        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )

        last_pos = attention_mask.sum(dim=1) - 1
        logits = out.logits[
            torch.arange(len(batch), device=DEVICE),
            last_pos,
        ].float()

        logp = F.log_softmax(logits.double(), dim=-1).cpu()

        for i, item in enumerate(batch):
            result[item["order_hash"]] = logp[i].contiguous()

        done = min(start + batch_size, len(items))
        print(
            f"    full distribution: {done}/{len(items)}",
            flush=True,
        )

    return result


def verify_family_structure(tokenizer, family_items):
    expected_counter = Counter(
        make_base_multiset(family_items[0]["counts"])
    )

    task_char_lengths = set()
    token_lengths = set()

    for item in family_items:
        if Counter(item["order"]) != expected_counter:
            raise AssertionError(
                f"Evidence multiset mismatch: {item['order_hash']}"
            )

        task = make_task(item["order"])
        task_char_lengths.add(len(task))

        ids, _ = prefix_ids_for(
            tokenizer, item["order"], "structured_only"
        )
        token_lengths.add(len(ids))

    return {
        "same_evidence_multiset": True,
        "task_char_lengths": sorted(task_char_lengths),
        "baseline_prefix_token_lengths": sorted(token_lengths),
        "same_task_char_length": len(task_char_lengths) == 1,
        "same_baseline_token_length": len(token_lengths) == 1,
    }


def baseline_pair_candidates(
    rows,
    max_dlo,
    max_dh,
    p_min,
    p_max,
):
    eligible = [
        r
        for r in rows
        if r["rank"] == 1
        and p_min <= r["p_correct"] <= p_max
    ]

    eligible.sort(key=lambda r: r["logodds"])

    candidates = []

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

                candidates.append({
                    "a": a,
                    "b": b,
                    "delta_logodds": dlo,
                    "delta_entropy": dh,
                    "delta_p": dp,
                    "delta_p_pp": dp * 100.0,
                })

            j += 1

    return candidates


def js_divergence_from_logp(logp_a, logp_b):
    a = logp_a.double()
    b = logp_b.double()

    log_m = torch.logaddexp(a, b) - math.log(2.0)

    pa = a.exp()
    pb = b.exp()

    kl_a = torch.sum(pa * (a - log_m))
    kl_b = torch.sum(pb * (b - log_m))

    js = 0.5 * (kl_a + kl_b)
    return max(0.0, float(js.item()))


def total_variation_from_logp(logp_a, logp_b):
    a = logp_a.double().exp()
    b = logp_b.double().exp()
    return float(
        (0.5 * torch.sum(torch.abs(a - b))).item()
    )


def max_probability_gap(logp_a, logp_b):
    a = logp_a.double().exp()
    b = logp_b.double().exp()
    return float(
        torch.max(torch.abs(a - b)).item()
    )


def add_distribution_metrics(candidates, logp_map):
    out = []

    for i, c in enumerate(candidates, 1):
        ha = c["a"]["order_hash"]
        hb = c["b"]["order_hash"]

        la = logp_map[ha]
        lb = logp_map[hb]

        x = dict(c)
        x["baseline_js_nats"] = js_divergence_from_logp(
            la, lb
        )
        x["baseline_tv"] = total_variation_from_logp(
            la, lb
        )
        x["baseline_max_prob_gap"] = max_probability_gap(
            la, lb
        )

        out.append(x)

        if i % 250 == 0 or i == len(candidates):
            print(
                f"    pair JS metrics: {i}/{len(candidates)}",
                flush=True,
            )

    out.sort(
        key=lambda x: (
            x["baseline_js_nats"],
            x["baseline_tv"],
            x["delta_logodds"],
            x["delta_entropy"],
        )
    )

    return out


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


def pair_condition_token_audit(tokenizer, pair):
    out = {}

    for cond in CONDITIONS:
        a_ids, _ = prefix_ids_for(
            tokenizer, pair["a"]["order"], cond
        )
        b_ids, _ = prefix_ids_for(
            tokenizer, pair["b"]["order"], cond
        )

        out[cond] = {
            "a": len(a_ids),
            "b": len(b_ids),
            "same": len(a_ids) == len(b_ids),
        }

    return out


def pearson(xs, ys):
    if len(xs) < 2:
        return float("nan")

    mx = statistics.fmean(xs)
    my = statistics.fmean(ys)

    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]

    den = math.sqrt(
        sum(x * x for x in dx)
        * sum(y * y for y in dy)
    )

    if den == 0:
        return float("nan")

    return sum(
        x * y for x, y in zip(dx, dy)
    ) / den


def summarize_pair(pair, profile_by_key):
    a0 = pair["a"]
    b0 = pair["b"]

    ha = a0["order_hash"]
    hb = b0["order_hash"]

    avec = []
    bvec = []
    condition_rows = []

    flips_a = 0
    flips_b = 0

    for cond in CONDITIONS[1:]:
        a = profile_by_key[(ha, cond)]
        b = profile_by_key[(hb, cond)]

        da = a["logodds"] - a0["logodds"]
        db = b["logodds"] - b0["logodds"]

        avec.append(da)
        bvec.append(db)

        flips_a += int(a["rank"] != 1)
        flips_b += int(b["rank"] != 1)

        condition_rows.append({
            "condition": cond,
            "a_dlo": da,
            "b_dlo": db,
            "abs_profile_diff": abs(da - db),
            "a_p_pct": a["p_correct_pct"],
            "b_p_pct": b["p_correct_pct"],
            "a_rank": a["rank"],
            "b_rank": b["rank"],
        })

    diffs = [
        abs(a - b)
        for a, b in zip(avec, bvec)
    ]

    sq = [
        (a - b) ** 2
        for a, b in zip(avec, bvec)
    ]

    return {
        "family": a0["family"],
        "counts": a0["counts"],
        "a_hash": ha,
        "b_hash": hb,

        "baseline_delta_logodds": pair["delta_logodds"],
        "baseline_delta_entropy": pair["delta_entropy"],
        "baseline_delta_p_pp": pair["delta_p_pp"],

        "baseline_js_nats": pair["baseline_js_nats"],
        "baseline_tv": pair["baseline_tv"],
        "baseline_max_prob_gap": pair["baseline_max_prob_gap"],

        "profile_mae": statistics.fmean(diffs),
        "profile_rmse": math.sqrt(
            statistics.fmean(sq)
        ),
        "profile_pearson": pearson(avec, bvec),

        "flip_count_a": flips_a,
        "flip_count_b": flips_b,

        "conditions": condition_rows,
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--n-per-family",
        type=int,
        default=768,
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=20260819,
    )

    # Old scalar gates.
    ap.add_argument(
        "--max-dlo",
        type=float,
        default=0.01,
    )
    ap.add_argument(
        "--max-dh",
        type=float,
        default=0.02,
    )
    ap.add_argument(
        "--p-min",
        type=float,
        default=0.80,
    )
    ap.add_argument(
        "--p-max",
        type=float,
        default=0.995,
    )

    ap.add_argument(
        "--pairs-per-family",
        type=int,
        default=4,
    )

    ap.add_argument(
        "--outdir",
        default="full_distribution_validation",
    )

    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    baseline_path = outdir / "baselines.jsonl"
    pairs_path = outdir / "selected_pairs.json"
    profile_path = outdir / "profiles.jsonl"
    final_path = outdir / "final_summary.json"

    print("=" * 90)
    print("STANDALONE FULL-DISTRIBUTION RESPONSE-GEOMETRY TEST")
    print("=" * 90)
    print()
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

    # ========================================================
    # 1) Generate exact controlled states.
    # ========================================================

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

        audit = verify_family_structure(
            tokenizer, items
        )

        structural_audits[fam] = audit

        print(
            f"FAMILY {fam}: "
            f"sampled {len(items)} / {total}"
        )
        print("  audit:", audit)

        if not audit["same_task_char_length"]:
            raise AssertionError(
                f"Character-length mismatch in family {fam}"
            )

        all_items.extend(items)

    # ========================================================
    # 2) Baseline score.
    # ========================================================

    old_base = load_jsonl(baseline_path)

    base_map = {
        (r["family"], r["order_hash"]): r
        for r in old_base
    }

    todo = [
        x
        for x in all_items
        if (x["family"], x["order_hash"])
        not in base_map
    ]

    print()
    print(
        f"Baseline checkpoint: "
        f"{len(base_map)} existing, {len(todo)} to score"
    )

    todo_by_family = defaultdict(list)

    for item in todo:
        todo_by_family[item["family"]].append(item)

    for fam, items in todo_by_family.items():
        print(f"\nBASELINE family {fam}")

        rows = score_items(
            model,
            tokenizer,
            items,
            "structured_only",
            args.batch_size,
            answer_ids,
            correct_id,
        )

        append_jsonl(
            baseline_path,
            rows,
        )

        for r in rows:
            base_map[
                (r["family"], r["order_hash"])
            ] = r

    # ========================================================
    # 3) Baseline-only scalar shortlist.
    # ========================================================

    rows_by_family = defaultdict(list)

    for r in base_map.values():
        rows_by_family[r["family"]].append(r)

    item_by_family_hash = {
        (x["family"], x["order_hash"]): x
        for x in all_items
    }

    selected = []
    matching_landscape = {}

    print()
    print("=" * 90)
    print("FULL-DISTRIBUTION BASELINE MATCHING")
    print("No perturbation has been scored yet.")
    print("=" * 90)

    for fam in sorted(rows_by_family):
        rows = rows_by_family[fam]

        candidates = baseline_pair_candidates(
            rows,
            args.max_dlo,
            args.max_dh,
            args.p_min,
            args.p_max,
        )

        print(f"\nFAMILY {fam}")
        print(
            f"  scalar-matched candidate pairs: "
            f"{len(candidates)}"
        )

        if not candidates:
            matching_landscape[fam] = {
                "scalar_candidates": 0,
                "full_distribution_states": 0,
                "selected_pairs": 0,
            }
            continue

        needed_hashes = set()

        for c in candidates:
            needed_hashes.add(
                c["a"]["order_hash"]
            )
            needed_hashes.add(
                c["b"]["order_hash"]
            )

        needed_items = [
            item_by_family_hash[(fam, h)]
            for h in sorted(needed_hashes)
        ]

        print(
            f"  states re-scored for complete distribution: "
            f"{len(needed_items)}"
        )

        logp_map = score_full_logprobs(
            model,
            tokenizer,
            needed_items,
            args.batch_size,
            answer_ids,
        )

        candidates = add_distribution_metrics(
            candidates,
            logp_map,
        )

        chosen_raw = choose_disjoint(
            candidates,
            args.pairs_per_family,
        )

        clean = []

        for c in chosen_raw:
            audit = pair_condition_token_audit(
                tokenizer, c
            )

            if all(
                x["same"]
                for x in audit.values()
            ):
                x = dict(c)
                x["token_audit"] = audit
                clean.append(x)

        selected.extend(clean)

        js_values = [
            c["baseline_js_nats"]
            for c in candidates
        ]

        matching_landscape[fam] = {
            "scalar_candidates": len(candidates),
            "full_distribution_states": len(needed_items),
            "min_js_nats": min(js_values),
            "median_js_nats": statistics.median(js_values),
            "selected_pairs": len(clean),
        }

        print(
            f"  minimum candidate JS: "
            f"{min(js_values):.12g} nats"
        )

        for i, c in enumerate(clean, 1):
            print(
                f"    {i}. "
                f"{c['a']['order_hash']} vs "
                f"{c['b']['order_hash']} | "
                f"JS={c['baseline_js_nats']:.12g} "
                f"TV={c['baseline_tv']:.12g} "
                f"max|Δp|={c['baseline_max_prob_gap']:.12g} "
                f"ΔLO={c['delta_logodds']:.8f} "
                f"ΔH={c['delta_entropy']:.8f}"
            )

        del logp_map

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ========================================================
    # 4) Freeze selection BEFORE perturbations.
    # ========================================================

    with open(
        pairs_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "model": MODEL,
                "selection_uses_perturbation_results": False,
                "selection_metric": (
                    "minimum complete-vocabulary "
                    "Jensen-Shannon divergence "
                    "within scalar-matched candidates"
                ),
                "matching_landscape": matching_landscape,
                "pairs": selected,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 90)
    print("PAIR SELECTION FROZEN")
    print("=" * 90)
    print(
        f"{len(selected)} pairs saved to {pairs_path}"
    )
    print(
        "Perturbations begin only after this point."
    )

    if not selected:
        print("No pairs survived. Stopping.")
        return

    # ========================================================
    # 5) Perturbation scoring.
    # ========================================================

    selected_items = {}

    for pair in selected:
        for side in ("a", "b"):
            r = pair[side]

            selected_items[
                (r["family"], r["order_hash"])
            ] = {
                "family": r["family"],
                "counts": r["counts"],
                "order": r["order"],
                "order_hash": r["order_hash"],
            }

    old_prof = load_jsonl(profile_path)

    prof_map = {
        (r["order_hash"], r["condition"]): r
        for r in old_prof
    }

    selected_by_family = defaultdict(list)

    for item in selected_items.values():
        selected_by_family[
            item["family"]
        ].append(item)

    for cond in CONDITIONS[1:]:
        print(f"\nPROFILE CONDITION {cond}")

        for fam, fam_items in selected_by_family.items():
            todo_cond = [
                item
                for item in fam_items
                if (
                    item["order_hash"],
                    cond,
                ) not in prof_map
            ]

            if not todo_cond:
                continue

            rows = score_items(
                model,
                tokenizer,
                todo_cond,
                cond,
                args.batch_size,
                answer_ids,
                correct_id,
            )

            append_jsonl(
                profile_path,
                rows,
            )

            for r in rows:
                prof_map[
                    (r["order_hash"], r["condition"])
                ] = r

    # ========================================================
    # 6) Response-geometry summary.
    # ========================================================

    summaries = [
        summarize_pair(pair, prof_map)
        for pair in selected
    ]

    summaries.sort(
        key=lambda x: x["baseline_js_nats"]
    )

    print()
    print("=" * 90)
    print("FINAL FULL-DISTRIBUTION VALIDATION")
    print("=" * 90)

    for i, s in enumerate(summaries, 1):
        print(
            f"{i:02d}. "
            f"family={s['family']}  "
            f"JS={s['baseline_js_nats']:.12g}  "
            f"TV={s['baseline_tv']:.12g}  "
            f"ΔLO={s['baseline_delta_logodds']:.8f}  "
            f"profile_MAE={s['profile_mae']:.4f}  "
            f"RMSE={s['profile_rmse']:.4f}  "
            f"r={s['profile_pearson']:+.4f}  "
            f"flips={s['flip_count_a']}/"
            f"{s['flip_count_b']}"
        )

    js_vals = [
        s["baseline_js_nats"]
        for s in summaries
    ]

    tv_vals = [
        s["baseline_tv"]
        for s in summaries
    ]

    dlos = [
        s["baseline_delta_logodds"]
        for s in summaries
    ]

    maes = [
        s["profile_mae"]
        for s in summaries
    ]

    print()
    print("AGGREGATE")
    print(
        f"  N pairs:               "
        f"{len(summaries)}"
    )
    print(
        f"  median baseline JS:     "
        f"{statistics.median(js_vals):.12g} nats"
    )
    print(
        f"  median baseline TV:     "
        f"{statistics.median(tv_vals):.12g}"
    )
    print(
        f"  median baseline ΔLO:    "
        f"{statistics.median(dlos):.8f}"
    )
    print(
        f"  mean profile MAE:       "
        f"{statistics.fmean(maes):.4f}"
    )
    print(
        f"  median profile MAE:     "
        f"{statistics.median(maes):.4f}"
    )
    print(
        f"  max profile MAE:        "
        f"{max(maes):.4f}"
    )

    if len(summaries) >= 2:
        print(
            f"  corr(JS, MAE):          "
            f"{pearson(js_vals, maes):+.4f}"
        )
        print(
            f"  corr(TV, MAE):          "
            f"{pearson(tv_vals, maes):+.4f}"
        )
        print(
            f"  corr(ΔLO, MAE):         "
            f"{pearson(dlos, maes):+.4f}"
        )

    final = {
        "model": MODEL,
        "device": DEVICE,
        "dtype": "float32",
        "attn_implementation": "eager",
        "use_cache": False,
        "seed": args.seed,
        "n_per_family": args.n_per_family,
        "count_families": [
            list(x)
            for x in COUNT_FAMILIES
        ],
        "conditions": CONDITIONS,
        "structural_audits": structural_audits,
        "matching_landscape": matching_landscape,
        "pair_summaries": summaries,
        "aggregate": {
            "n_pairs": len(summaries),
            "median_baseline_js_nats":
                statistics.median(js_vals),
            "median_baseline_tv":
                statistics.median(tv_vals),
            "median_baseline_delta_logodds":
                statistics.median(dlos),
            "mean_profile_mae":
                statistics.fmean(maes),
            "median_profile_mae":
                statistics.median(maes),
            "max_profile_mae":
                max(maes),
            "pearson_js_vs_mae":
                (
                    pearson(js_vals, maes)
                    if len(summaries) >= 2
                    else None
                ),
            "pearson_tv_vs_mae":
                (
                    pearson(tv_vals, maes)
                    if len(summaries) >= 2
                    else None
                ),
        },
    }

    with open(
        final_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            final,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("Saved:")
    print(" ", baseline_path)
    print(" ", pairs_path)
    print(" ", profile_path)
    print(" ", final_path)


if __name__ == "__main__":
    main()
