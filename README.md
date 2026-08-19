# Prompt-State Response Geometry

This is not a paper.

It is a small hobby project that started with a simple question:

> Can we make an LLM recall factual information more reliably just by changing the prompt?

A few days later, we ended up somewhere we did not expect.

The main observation is:

> **Nearly identical local confidence does not imply nearly identical perturbational response geometry.**

We are sharing the code and raw outputs because the experiment is small enough to inspect directly, and because we would rather let other people reproduce, break, extend, or explain it than pretend we have the final answer.

We are **not** claiming a new transformer mechanism, a hallucination detector, or a universal property of language models.

## The core control

The final experiment changes only the **order** of an identical evidence multiset.

Within each evidence family:

- the evidence values are identical,
- their counts are identical,
- the literal evidence lines are identical,
- task character length is identical,
- baseline token length is identical,
- and matched pairs are also audited condition-by-condition for equal token length.

Most importantly:

> **Pairs are selected using baseline confidence statistics before any perturbation results are scored.**

The later perturbation outcomes therefore cannot influence which pairs are chosen.

That outcome-blind selection is the main reason we consider the final experiment more useful than the earlier exploratory versions.

## The basic idea

Suppose a model is currently predicting some token.

We can describe that prediction with local statistics such as:

- probability of the target token,
- log-odds,
- entropy,
- rank.

We became interested in a different question:

> If two prompt states look almost identical by those local confidence statistics, do they also react similarly when the surrounding context changes a little?

In this controlled Granite experiment, often they did not.

A mental model we found useful is:

- **confidence is a snapshot**
- **perturbation response is the nearby terrain**

Two points can have the same elevation without having the same landscape around them.

---

# Where this started: Gemma

The first version of the broader phenomenon showed up while experimenting with **Gemma 4 26B-A4B**.

At the time, we were using a llama.cpp / GGUF setup and looking at factual recall boundaries.

One example was the creator of Rust:

> Graydon Hoare

The token path contained a boundary around:

```text
Graydon Ho | are
```

The model could be extremely stable on the earlier part of the name while the probability of the final continuation changed dramatically under companion prompts.

For example, the probability of continuing from `Graydon` to `Ho` stayed near saturation, while the probability of continuing from `Graydon Ho` to `are` could move from very high confidence to single-digit probabilities depending on the surrounding prompt state.

At first we suspected almost everything:

- sampling,
- llama.cpp,
- quantization,
- prompt templates,
- KV/cache behavior,
- tokenization,
- or just a weird model-specific artifact.

So we stopped treating the Gemma result as strong evidence by itself and moved to a cleaner setup.

The Gemma observation and the final Granite experiment are **not** an apples-to-apples replication. They differ in architecture, model family, backend, model size, and other details.

We treat Gemma only as convergent evidence that the broader prompt-state sensitivity we saw is not obviously unique to the final Granite setup.

The exact random-order matched experiment in this repository was performed on Granite.

---

# Controlled Granite setup

We switched to:

```text
ibm-granite/granite-4.1-3b
```

using raw Hugging Face Transformers with:

```text
float32
attn_implementation="eager"
use_cache=False
no sampling
no GGUF
no quantization
teacher-forced next-token scoring
```

This removed several moving parts from the earlier setup.

---

# The synthetic task

We created a temporary fictional record task using four possible values:

```text
Igor Sysoev
Igor Sokolov
Igor Smith
Igor Petrov
```

The target continuation is:

```text
Igor | Sy
```

We teacher-force:

```text
" Igor"
```

and score:

```text
" Sy"
```

The evidence consists of repeated fictional records such as:

```text
K7 -> Igor Sysoev
K7 -> Igor Smith
K7 -> Igor Sysoev
K7 -> Igor Petrov
...
```

Within one evidence family, only the **order of these evidence lines** changes.

No record numbers are used in the final validation, so reordering does not silently change the literal record contents.

---

# Random-order validation

The final experiment uses five evidence-count families:

```text
6 / 1 / 4 / 1
4 / 3 / 1 / 2
5 / 0 / 5 / 4
5 / 4 / 1 / 2
6 / 5 / 3 / 0
```

For each family, we sample:

```text
768 unique random permutations
```

for a total of:

```text
3840 baseline prompt states
```

The script performs structural checks before scoring.

---

# Outcome-blind pair selection

We do **not** inspect perturbation results and then choose interesting-looking pairs.

Pairs are selected from baseline measurements only.

The default criteria are:

```text
rank = 1
P(correct) between 0.80 and 0.995
|delta log-odds| <= 0.01
|delta entropy| <= 0.02
```

Pair selection happens before perturbation scoring, and selected pairs are disjoint within each family.

The frozen selection is saved in:

```text
results/selected_pairs.json
```

---

# Perturbations

After the matched pairs are frozen, the same 12 companion perturbations are applied to both states in every pair:

```text
repeat LUMA
repeat NOVA
repeat KITE
repeat 4827

copy LUMA
copy NOVA
copy KITE
copy 4827

write LUMA
write NOVA
write KITE
write 4827
```

These are small companion tasks that do not change the fictional K7 evidence itself.

For each state, we measure how much the target token's log-odds move relative to that state's own baseline.

This gives a 12-dimensional response vector:

```text
[dLO_1, dLO_2, ..., dLO_12]
```

For each matched pair, the main comparison is:

```text
profile MAE
```

the mean absolute condition-by-condition difference between the two response vectors.

---

# Final result

The included random-order validation produced:

```text
N matched pairs:             20

median baseline delta LO:    0.00079255

mean profile MAE:            2.9246
median profile MAE:          2.2979
maximum profile MAE:        10.0046
```

The matched baselines can be extremely close.

For example, one pair had approximately:

```text
baseline P(correct):
93.613813%
93.614706%

delta log-odds:
0.0001493
```

yet its perturbation profile MAE was:

```text
2.8936
```

Another pair had a baseline log-odds difference of only about:

```text
0.00107
```

but a profile MAE of roughly:

```text
10.00
```

In several pairs, one state lost top-1 status under many or all perturbations while its baseline-matched partner did not.

So, in this setup:

> **Matching the local prediction very closely does not guarantee matching how that prediction responds to nearby context changes.**

---

# A simple within-sample null

A useful criticism of the first README was: "2.92 compared to what?"

We can answer part of that without another model run.

The selected set contains 40 states: 8 states from each of 5 evidence families.

We compared the 20 outcome-blind confidence-matched pairs against the other within-family pairings among those same 40 states.

```text
                         matched pairs    other within-family pairs
N                              20                 120

mean profile MAE             2.9246              3.0348
median profile MAE           2.2979              2.1155
```

In this selected set, very tight confidence matching did **not** make the perturbation-response profiles dramatically more similar than the other within-family pairings.

We do **not** interpret this as "confidence contains no information." This is a descriptive, conditioned comparison among the 40 states that were already selected for profile scoring, not a population-level statistical test over all 3840 baseline states.

It does make the narrow claim easier to state:

> **Matching these local confidence statistics is not enough to determine the perturbational response profile.**

---

# Log-odds and probability saturation

An earlier version of this project computed log-odds from clamped probabilities. That could become numerically misleading near probability saturation.

The final experiment does **not** use that metric.

Log-odds are computed directly from logits:

```text
target_logit - logsumexp(all_other_logits)
```

with the metric calculation performed in float64.

Some perturbed states still approach probability saturation, so large log-odds can naturally occur even without the old numerical bug.

As a descriptive sensitivity check, we clipped perturbed log-odds at the magnitude corresponding to **99.99% confidence**.

```text
original mean profile MAE:   2.9246
clipped mean profile MAE:    2.7263
```

The effect became somewhat smaller, but it did not disappear.

This clipping check is an offline analysis of the included outputs; it is not part of pair selection.

---

# What we think this means

We do **not** think confidence is useless.

In earlier exploratory experiments, confidence and sensitivity were often correlated in the expected direction: higher-confidence predictions were generally more stable than uncertain ones.

The narrower observation here is:

> **Local confidence statistics do not appear to uniquely determine local perturbational behavior.**

Two states can look almost identical according to:

```text
P(correct)
log-odds
entropy
rank
```

while still having very different response profiles under the same perturbation set.

A single local confidence snapshot may therefore be an incomplete description of a prompt state if the thing we care about is how that state behaves under nearby context changes.

---

# Relation to prompt sensitivity

Prompt sensitivity itself is not new.

It is already well known that:

- prompt wording matters,
- few-shot example order matters,
- context order matters,
- semantically similar prompts can produce different outputs.

Our experiment asks a narrower question.

Instead of only asking:

> Does changing the prompt change the prediction?

we ask:

> If two prompt states already look almost identical at the current prediction boundary, do they also respond similarly to the same later perturbations?

Our controlled Granite result suggests that the answer can be no.

We intentionally do not make a stronger novelty claim than that.

---

# What this does NOT show

This repository does not demonstrate:

- a hallucination detector,
- a universal property of all LLMs,
- a new transformer mechanism,
- that evidence order always matters,
- that one ordering strategy is globally better,
- that confidence is useless,
- that factual memory is uniquely fragile,
- that post-training causes the effect.

The exact internal mechanism is unknown.

Possible explanations could involve position, attention history, contextual representations, residual-stream state, or something else entirely.

We have not isolated that mechanism.

---

# Important limitations

The final controlled experiment is still narrow.

### One controlled model

The exact random-order matched experiment was performed on:

```text
ibm-granite/granite-4.1-3b
```

Gemma 4 26B-A4B showed the broader phenomenon earlier, but the exact random-order design has not been replicated there.

### One synthetic task

The final experiment uses:

```text
Igor Sysoev
Igor Sokolov
Igor Smith
Igor Petrov
```

Other token geometries or synthetic tasks may behave differently.

### One perturbation family

The 12 perturbations are hand-designed.

They are not a random sample from all possible companion prompts.

### Matching summary statistics, not the full distribution

Pairs are matched on target probability/log-odds, entropy, and rank.

We do **not** match the complete baseline next-token distribution.

Two states can therefore have similar target confidence summaries while differing in the detailed distribution over competitors.

A stricter future control could match or constrain full-distribution divergence directly.

### Descriptive result

This repository focuses on demonstrating and reproducing the observation.

We are not presenting formal inferential statistics or claiming a population-level effect size.

---

# Why we stopped testing

There is always another possible control.

We could test:

- another model family,
- another name set,
- another synthetic task,
- random perturbation families,
- different token boundaries,
- hidden-state similarity,
- full next-token distribution matching,
- larger permutation ensembles.

At some point, we decided the more useful thing was to release the smallest reproducible version and let other people attack it.

If the observation disappears under a better control, that is useful.

If it replicates elsewhere, that is useful too.

---

# Reproducing the experiment

The repository intentionally contains one main experiment script:

```text
experiment.py
```

Run:

```bash
python experiment.py
```

The default run samples:

```text
768 permutations per evidence family
```

and writes its outputs to:

```text
random_order_validation/
```

The script supports checkpointing, so interrupted runs can be resumed.

For a larger permutation search:

```bash
python experiment.py --n-per-family 1500
```

---

# Environment used for the included results

The included outputs were generated with:

```text
Python:       3.14.4
PyTorch:      2.12.0+rocm7.14.0
Transformers: 5.14.1
HIP:          7.14.60850
GPU:          AMD Radeon RX 7900 XT
```

Model:

```text
ibm-granite/granite-4.1-3b
```

Model settings:

```text
float32
eager attention
use_cache=False
```

ROCm PyTorch exposes the GPU through the normal:

```python
device = "cuda"
```

interface.

---

# Repository contents

```text
README.md
LICENSE
requirements.txt
experiment.py
.gitignore

results/
    final_summary.json
    selected_pairs.json
    baselines.jsonl
    profiles.jsonl
    run_output.txt
```

`baselines.jsonl` contains the baseline permutation scan.

`selected_pairs.json` contains the frozen outcome-blind matched-pair selection.

`profiles.jsonl` contains perturbation scoring.

`final_summary.json` contains the final pair-level metrics and structural audits.

`run_output.txt` is the raw terminal output from the included run.

---

# A note about discussion and responses

This is a hobby project, not a maintained research program or a paper we intend to defend indefinitely.

We are sharing the code, raw outputs, methodology, and limitations so the experiment can be inspected without needing to trust us personally.

We may not respond to comments, issues, messages, requests for debate, or every proposed follow-up experiment.

A lack of response should **not** be interpreted as agreement, disagreement, or as a statement about the validity of a criticism.

If you find a problem with the experiment, that is useful. The most useful way to challenge the result is to reproduce it, modify the code, run a stronger control, or show a counterexample.

There is no expectation that this repository will become an ongoing research project. We may update it if something seems worth adding, or we may simply leave it as-is.

---

# AI collaboration

This project was developed collaboratively between the repository owner and ChatGPT.

The original direction came from the repo owner:

> Can prompt design make an LLM recall information more reliably?

From there, the project changed direction repeatedly as unexpected behavior appeared.

The repo owner ran the models locally and steered the investigation by questioning experimental choices, spotting setup problems, rejecting rabbit holes, and pushing for cleaner controls — including moving away from llama.cpp when backend and sampling concerns became relevant.

Most of the experimental code and most of the detailed quantitative analysis were produced by **ChatGPT (OpenAI)** during interactive sessions.

ChatGPT also wrote this README from the experiment history, code, and recorded results, with review and direction from the repo owner.

So this repository should not be read as:

> "A human researcher wrote everything and occasionally used AI autocomplete."

That would not be accurate.

A better description is:

> **We explored the problem interactively: the repo owner provided the initial idea, local execution, skepticism, and steering; ChatGPT provided most of the code, quantitative analysis, and experiment iteration.**

We are explicitly disclosing this because there is no reason to hide it, especially in a project about language models.

The code and raw outputs are included so nobody has to trust either of us.

Please reproduce it, break it, simplify it, or find a better explanation.

You do not need our participation to do any of that.

---

# License

This repository is released under **The Unlicense**.

The intent is simple:

use it, copy it, modify it, fork it, extend it, or throw it away.

No permission is needed from us.

The licenses and terms of the model, PyTorch, Transformers, and other third-party dependencies remain their own.
