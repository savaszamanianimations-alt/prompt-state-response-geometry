import argparse
import json
import math
import statistics
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# Standalone repeatability / numerical-noise control
# Requires ONLY the selected_pairs.json produced by the previous
# standalone full-distribution experiment.
# No repo files are needed.
# ============================================================

MODEL = "ibm-granite/granite-4.1-3b"
DEVICE = "cuda"

VALUES = (
    "Igor Sysoev",
    "Igor Sokolov",
    "Igor Smith",
    "Igor Petrov",
)
ANSWER_TEXT = " Igor Sysoev"

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

    ids = tokenizer.encode(rendered, add_special_tokens=False)
    return ids


def answer_boundary_ids(tokenizer):
    ids = tokenizer.encode(ANSWER_TEXT, add_special_tokens=False)
    toks = [tokenizer.decode([x]) for x in ids]

    if len(ids) < 2:
        raise RuntimeError(f"Unexpected answer tokenization: {toks}")

    return ids, toks


def target_metrics(logits, correct_id):
    z = logits.double()

    correct = z[:, correct_id]

    other_z = z.clone()
    other_z[:, correct_id] = -torch.inf
    others = torch.logsumexp(other_z, dim=-1)

    logodds = correct - others

    logp = F.log_softmax(z, dim=-1)
    p = logp[:, correct_id].exp()

    return p, logodds


@torch.inference_mode()
def score_condition(
    model,
    tokenizer,
    items,
    condition,
    batch_size,
    answer_ids,
    correct_id,
):
    force_first = answer_ids[0]
    out_rows = []

    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]

        seqs = []
        raw_lengths = []

        for item in batch:
            prefix = prefix_ids_for(
                tokenizer,
                item["order"],
                condition,
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

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )

        last_pos = attention_mask.sum(dim=1) - 1

        logits = outputs.logits[
            torch.arange(len(batch), device=DEVICE),
            last_pos,
        ].float()

        p, lo = target_metrics(logits, correct_id)

        p = p.cpu().tolist()
        lo = lo.cpu().tolist()

        for i, item in enumerate(batch):
            out_rows.append({
                "family": item["family"],
                "order_hash": item["order_hash"],
                "condition": condition,
                "p_correct": float(p[i]),
                "logodds": float(lo[i]),
            })

    return out_rows


def profile_for_state(run_map, order_hash):
    base = run_map[(order_hash, "structured_only")]["logodds"]

    return [
        run_map[(order_hash, cond)]["logodds"] - base
        for cond in CONDITIONS[1:]
    ]


def mae(a, b):
    return statistics.fmean(
        abs(x - y)
        for x, y in zip(a, b)
    )


def rmse(a, b):
    return math.sqrt(
        statistics.fmean(
            (x - y) ** 2
            for x, y in zip(a, b)
        )
    )


def max_abs(a, b):
    return max(
        abs(x - y)
        for x, y in zip(a, b)
    )


def pearson(xs, ys):
    if len(xs) < 2:
        return float("nan")

    mx = statistics.fmean(xs)
    my = statistics.fmean(ys)

    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]

    den = math.sqrt(
        sum(x*x for x in dx)
        * sum(y*y for y in dy)
    )

    if den == 0:
        return float("nan")

    return sum(
        x*y for x, y in zip(dx, dy)
    ) / den


def load_selected_pairs(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pairs = data["pairs"]

    items = {}
    for pair in pairs:
        for side in ("a", "b"):
            r = pair[side]

            items[r["order_hash"]] = {
                "family": r["family"],
                "order_hash": r["order_hash"],
                "order": r["order"],
            }

    return pairs, list(items.values())


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--pairs",
        default="full_distribution_validation/selected_pairs.json",
        help="selected_pairs.json from the previous experiment",
    )
    ap.add_argument(
        "--repeats",
        type=int,
        default=3,
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )
    ap.add_argument(
        "--out",
        default="repeatability_summary.json",
    )

    args = ap.parse_args()

    pairs_path = Path(args.pairs)

    if not pairs_path.exists():
        raise FileNotFoundError(
            f"Could not find {pairs_path}\n"
            "Put this script in the same directory where "
            "full_distribution_validation/ exists, or pass:\n"
            "  python repeatability_test.py --pairs /path/to/selected_pairs.json"
        )

    pairs, items = load_selected_pairs(pairs_path)

    print("=" * 88)
    print("REPEATABILITY / NUMERICAL-NOISE CONTROL")
    print("=" * 88)
    print(f"Pairs:  {len(pairs)}")
    print(f"States: {len(items)}")
    print(f"Runs:   {args.repeats}")
    print()

    print("Loading model...", flush=True)

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

    # --------------------------------------------------------
    # Run the exact same states / exact same conditions
    # multiple times.
    # --------------------------------------------------------

    runs = []

    for rep in range(args.repeats):
        print("=" * 88)
        print(f"RUN {rep + 1}/{args.repeats}")
        print("=" * 88)

        rows = []

        for ci, cond in enumerate(CONDITIONS, 1):
            print(
                f"  {ci:02d}/{len(CONDITIONS)}  {cond}",
                flush=True,
            )

            rows.extend(
                score_condition(
                    model,
                    tokenizer,
                    items,
                    cond,
                    args.batch_size,
                    answer_ids,
                    correct_id,
                )
            )

        run_map = {
            (r["order_hash"], r["condition"]): r
            for r in rows
        }

        runs.append(run_map)

    # --------------------------------------------------------
    # 1) Same-state repeatability
    # --------------------------------------------------------

    state_repeat_maes = []
    state_repeat_rmses = []
    state_repeat_maxes = []
    baseline_repeat_diffs = []

    repeat_details = []

    for item in items:
        h = item["order_hash"]

        profiles = [
            profile_for_state(run_map, h)
            for run_map in runs
        ]

        baselines = [
            run_map[(h, "structured_only")]["logodds"]
            for run_map in runs
        ]

        for i in range(len(runs)):
            for j in range(i + 1, len(runs)):
                m = mae(profiles[i], profiles[j])
                r = rmse(profiles[i], profiles[j])
                mx = max_abs(profiles[i], profiles[j])
                bd = abs(baselines[i] - baselines[j])

                state_repeat_maes.append(m)
                state_repeat_rmses.append(r)
                state_repeat_maxes.append(mx)
                baseline_repeat_diffs.append(bd)

                repeat_details.append({
                    "order_hash": h,
                    "family": item["family"],
                    "run_a": i + 1,
                    "run_b": j + 1,
                    "profile_mae": m,
                    "profile_rmse": r,
                    "profile_max_abs": mx,
                    "baseline_logodds_abs_diff": bd,
                })

    # --------------------------------------------------------
    # 2) Matched-pair response-profile difference, recomputed
    # independently inside every run.
    # --------------------------------------------------------

    pair_maes = []
    pair_rmses = []
    pair_details = []

    for run_index, run_map in enumerate(runs, 1):
        for pair_index, pair in enumerate(pairs, 1):
            ha = pair["a"]["order_hash"]
            hb = pair["b"]["order_hash"]

            pa = profile_for_state(run_map, ha)
            pb = profile_for_state(run_map, hb)

            m = mae(pa, pb)
            r = rmse(pa, pb)

            pair_maes.append(m)
            pair_rmses.append(r)

            pair_details.append({
                "run": run_index,
                "pair": pair_index,
                "family": pair["a"]["family"],
                "a_hash": ha,
                "b_hash": hb,
                "profile_mae": m,
                "profile_rmse": r,
            })

    # --------------------------------------------------------
    # 3) Stability of the matched-pair MAE itself across runs
    # --------------------------------------------------------

    per_pair_across_runs = defaultdict(list)

    for row in pair_details:
        key = (
            row["pair"],
            row["a_hash"],
            row["b_hash"],
        )
        per_pair_across_runs[key].append(row["profile_mae"])

    pair_mae_run_spreads = []

    for vals in per_pair_across_runs.values():
        if len(vals) >= 2:
            pair_mae_run_spreads.append(
                max(vals) - min(vals)
            )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    med_self = statistics.median(state_repeat_maes)
    max_self = max(state_repeat_maes)
    med_pair = statistics.median(pair_maes)

    ratio = (
        med_pair / med_self
        if med_self > 0
        else float("inf")
    )

    print()
    print("=" * 88)
    print("REPEATABILITY RESULT")
    print("=" * 88)

    print()
    print("SAME STATE, REPEATED FORWARD PASSES")
    print(
        f"  median baseline |ΔLO|:      "
        f"{statistics.median(baseline_repeat_diffs):.12g}"
    )
    print(
        f"  max baseline |ΔLO|:         "
        f"{max(baseline_repeat_diffs):.12g}"
    )
    print(
        f"  median self profile MAE:    "
        f"{med_self:.12g}"
    )
    print(
        f"  mean self profile MAE:      "
        f"{statistics.fmean(state_repeat_maes):.12g}"
    )
    print(
        f"  max self profile MAE:       "
        f"{max_self:.12g}"
    )
    print(
        f"  median self profile RMSE:   "
        f"{statistics.median(state_repeat_rmses):.12g}"
    )
    print(
        f"  max single-condition drift: "
        f"{max(state_repeat_maxes):.12g}"
    )

    print()
    print("MATCHED PAIRS, SAME RUN")
    print(
        f"  median pair profile MAE:    "
        f"{med_pair:.6f}"
    )
    print(
        f"  mean pair profile MAE:      "
        f"{statistics.fmean(pair_maes):.6f}"
    )
    print(
        f"  max pair profile MAE:       "
        f"{max(pair_maes):.6f}"
    )

    print()
    print("SIGNAL / NUMERICAL-NOISE SCALE")
    print(
        f"  median pair MAE / median self MAE: "
        f"{ratio:.3g}x"
    )

    if pair_mae_run_spreads:
        print(
            f"  median pair-MAE run-to-run spread: "
            f"{statistics.median(pair_mae_run_spreads):.12g}"
        )
        print(
            f"  max pair-MAE run-to-run spread:    "
            f"{max(pair_mae_run_spreads):.12g}"
        )

    # --------------------------------------------------------
    # Simple interpretation guardrail
    # --------------------------------------------------------

    print()
    print("INTERPRETATION")

    if med_self == 0:
        print(
            "  Exact repeatability at measured precision. "
            "Numerical noise does not explain the matched-pair effect."
        )
    elif med_pair >= 100 * med_self:
        print(
            "  Matched-pair response differences are >=100x larger "
            "than same-state repeat noise."
        )
    elif med_pair >= 10 * med_self:
        print(
            "  Matched-pair response differences are >=10x larger "
            "than same-state repeat noise."
        )
    else:
        print(
            "  Same-state numerical drift is large enough that it must "
            "be treated as a serious alternative explanation."
        )

    summary = {
        "model": MODEL,
        "dtype": "float32",
        "attn_implementation": "eager",
        "use_cache": False,
        "pairs_path": str(pairs_path),
        "n_pairs": len(pairs),
        "n_states": len(items),
        "repeats": args.repeats,
        "same_state_repeatability": {
            "median_baseline_abs_logodds_diff":
                statistics.median(baseline_repeat_diffs),
            "max_baseline_abs_logodds_diff":
                max(baseline_repeat_diffs),
            "median_profile_mae":
                med_self,
            "mean_profile_mae":
                statistics.fmean(state_repeat_maes),
            "max_profile_mae":
                max_self,
            "median_profile_rmse":
                statistics.median(state_repeat_rmses),
            "max_single_condition_abs_drift":
                max(state_repeat_maxes),
        },
        "matched_pairs": {
            "median_profile_mae":
                med_pair,
            "mean_profile_mae":
                statistics.fmean(pair_maes),
            "max_profile_mae":
                max(pair_maes),
        },
        "signal_to_repeat_noise_ratio":
            ratio,
        "median_pair_mae_run_to_run_spread":
            (
                statistics.median(pair_mae_run_spreads)
                if pair_mae_run_spreads
                else None
            ),
        "max_pair_mae_run_to_run_spread":
            (
                max(pair_mae_run_spreads)
                if pair_mae_run_spreads
                else None
            ),
        "repeat_details": repeat_details,
        "pair_details": pair_details,
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
