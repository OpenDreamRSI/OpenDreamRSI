# OpenDream-RSI

An open reproduction of **[Dream-RSI: Recursive Self-Improvement through Evolving
Worlds](https://www.dream-rsi.com)** (Zheng et al., 2026).

> *"An agent must dream to recursively self-improve. History is the world it
> dreams in."*

The paper's key insight is that you do not need a *learned* world model to
improve an exploration policy. A completed discovery run already records a tree
of exploration decisions together with their realized code-execution outcomes,
so replaying a different policy over it means traversing different branches, in
a different order, with different parallel groupings and stopping decisions —
and every outcome is already stored. The history is an **exact** simulator of
the region that was explored, and a single expensive online run supports
thousands of zero-execution-cost off-policy evaluations.

This repository implements that loop end to end:

| Stage | Paper | Module |
|---|---|---|
| ① Online Explore | policy `π_t` guides the discovery agent, producing tree `T_t` | [`dreamrsi/online.py`](dreamrsi/online.py) |
| ② Construct Replay Simulator | `T_t` frozen and served back as a world | [`dreamrsi/world.py`](dreamrsi/world.py) |
| ③ Dreaming-based Policy Improvement | `M` policy revisions replayed over the pool | [`dreamrsi/dreaming.py`](dreamrsi/dreaming.py) |
| outer RSI loop | `π_{t+1} = π_t^{m*}`, redeployed online | [`dreamrsi/rsi.py`](dreamrsi/rsi.py) |
| — | does any of it hold up? | [`dreamrsi/experiments/`](dreamrsi/experiments/) |

**The authors have not released their code** (as of writing, `zhengkid/Dream-RSI`
contains the paper and assets only). This is an independent reimplementation
from the paper text, so read [Faithfulness](#faithfulness) before drawing
conclusions from it.

---

## Quickstart

No API key, no CLI, no network — the whole loop runs offline against a mock
discovery agent and a cheap multi-modal search task.

```bash
python3 -m pip install -e ".[dev,tasks]"
```

Walk through the three stages one at a time:

```bash
python3 examples/quickstart.py
```

Run the controlled comparison from Section 4 — Dream-RSI against **Recursive
Fixed Exploration**, same discovery agent, evaluator, initialisation and
per-round budget, differing only in whether the exploration policy is allowed to
improve:

```bash
python3 -m dreamrsi.cli compare --config configs/synthetic.yaml
```

```
round      fixed best  fixed calls       dream best  dream calls
--------------------------------------------------------------------
    1      -19.768276           78      -19.768276           78
    2      -17.798798          156      -17.798798          156
    3      -15.199224          234      -15.199224          234
    4      -17.770778          312      -17.770778          249
    5      -28.581682          390      -40.713889          264
    6      -18.668184          468      -18.668184          297
--------------------------------------------------------------------
     fixed: best=-15.199224  calls=468
     dream: best=-15.199224  calls=297
```

Same final quality, **1.58× fewer discovery-agent calls**. Rounds 1–3 are
identical by construction: both methods deploy the same manually designed
policy, and Dream-RSI only diverges once dreaming has found a version that beats
the incumbent on the accumulated history.

**That is one seed, and the quality tie does not survive more of them.** Across
eight seeds the call saving holds (1.75× on average, cheaper on 7/8) but the
quality tie is specific to seed 0 — run
[`dreamrsi benchmark`](#across-seeds-the-saving-is-real-the-quality-tie-is-not)
before quoting the table above. These are numbers from the offline synthetic
task, not a reproduction of the paper's benchmarks — see
[Faithfulness](#faithfulness).

---

## The model

### Discovery trees

A rollout builds a rooted tree containing the task root `r` (the initial
workspace). The only atomic operation is `CONTINUE(v)`: resume the workspace at
`v` and generate exactly one evaluated child. Continuing from the root opens a
new **branch**; continuing from a non-root leaf **refines** the end of an
existing branch.

Because every non-root node has exactly one primary parent, the tree is a set of
chains hanging off the root — equivalently, an irregular `branch × attempt`
grid, which is exactly the environment the policy prompt describes. Cells are
addressed as `b{branch}a{attempt}`.

### Replay worlds

`ReplayWorld` (`question` in the prompts) reveals recorded outcomes:

```python
question.reset()
question.observed()          -> dict[str, Observation]   # revealed prefix only
question.legal_actions()     -> list[str]                # roots + opened-branch frontiers
question.legal_roots()       -> list[str]                # unopened roots only
question.opened_branches()   -> list[int]
question.meta(cell_id)       -> CellMeta                 # .branch .attempt .parent_id .seq .tags
question.probe_batch(cells, on_reveal=...) -> list[Observation]
question.baseline_score
question.max_parallelism
```

`LiveWorld` implements the identical interface; the only difference is that a
continuation *generates* a new outcome (one discovery-agent call plus one
evaluation) instead of revealing a recorded one. A batch of `k` cells costs one
decision round and `ceil(k / W)` effective sequential rounds in both phases.

Illegal batches — duplicates, more than `W` cells, a cell that was not legal
before the call, or a parent together with its child — raise rather than being
silently trimmed, so a policy that cheats fails loudly during dreaming.

### The replay objectives

Equation (1) of the paper, in [`objective.py`](dreamrsi/objective.py):

```
V_i^m  =  max_{v ∈ T_i} s_v   −   β₁·N_i^m   +   β₂·N_i^m / max(1, k_i^m)
          discovery quality     execution cost   parallelism bonus
```

and the `pareto` reward the policy-development prompt optimises:

```
pareto.reward = pareto.auc − λ · parallel_penalty
```

* **`attainment`** — per-trace, normalised so `0` reproduces the baseline and `1`
  reaches that frozen trace's own ceiling. Using the trace ceiling rather than an
  absolute target keeps worlds with very different score scales comparable.
* **`pareto.auc`** — the area under the **upper envelope** of the
  attainment-vs-probes curves across the whole beta sweep. This is what "rewards
  reaching high per-trace attainment with few total probes" means when applied to
  a sweep rather than one episode: a knob that genuinely buys attainment cheaply
  lifts the envelope at both ends, while a cosmetic knob leaves it equal to the
  single curve it always produces.
* **`parallel_penalty`** — the mean of
  `effective_sequential_rounds / total_probes` over the sweep. A serial policy
  sits near `1`; useful full batches approach `1/W`.
* **`non_degenerate`** — whether beta actually moved the attainment/work
  trade-off. Reported on every sweep, because the prompt explicitly warns that a
  sweep is only informative if it does.

### Policy selection never regresses

The offline phase evaluates `π_t^0 = π_t` alongside the `M` revisions on the
same frozen pool with the same beta grid. Because the incumbent is always in the
candidate set, the selected `π_{t+1} = π_t^{m*}` satisfies `V^{m*} ≥ V^0`: the
next online policy is never worse than the current one in average replay score
on `H_t`. `tests/test_rsi_loop.py::test_selection_never_regresses_below_the_incumbent`
asserts this on every round of a real run.

Note what that guarantee does *and does not* say. It is an **in-pool** statement:
the winner won on the worlds it was scored on. `π_{t+1}` is then deployed on
`T_{t+1}`, which did not exist when it was selected. Whether the guarantee
survives that is an empirical question, and [Evidence](#evidence) measures it.

---

## Evidence

Everything above *implements* Dream-RSI. This section *interrogates* it: three
offline commands ([`dreamrsi/experiments/`](dreamrsi/experiments/)) that test the
claims the method's economy rests on rather than merely exercising the code.
Every number below is reproducible on the synthetic task with no model access,
except the one subsection that needs a real model and says so.

### The simulator is exact — inside the region that was explored

```bash
dreamrsi fidelity --config configs/synthetic.yaml
```

Deploys each shipped policy live at three betas, freezes the tree, **writes it to
disk, loads it back** and replays the same policy over it:

```
              policy   beta   exact   rounds live/replay   probes live/replay      trace
-----------------------------------------------------------------------------------------
                pi_1   0.20    True                13/13                78/78      78/78
                pi_1   0.62    True                13/13                78/78      78/78
                pi_1   1.00    True                13/13                78/78      78/78
  adaptive_portfolio   0.20    True                  5/5                27/27      27/78
  adaptive_portfolio   0.62    True                13/13                78/78      78/78
  adaptive_portfolio   1.00    True                13/13                78/78      78/78
       uct_portfolio   0.20    True                  3/3                17/17      17/78
       uct_portfolio   0.62    True                  4/4                24/24      24/78
       uct_portfolio   1.00    True                  5/5                30/30      30/78
-----------------------------------------------------------------------------------------
exact replays: 9/9   (trace = recorded cells out of the 78-cell live grid)
```

Same batches in the same order, same revealed scores, same best score, same probe
count — through serialisation, so it fails if freezing, storing or serving a world
ever stops reproducing what the live run saw. **The stored history does not lie.**

The second half of the claim is weaker, and the same command measures it. A
replay world offers a strictly smaller menu than the live grid did: roots the
online run never opened cannot be opened in a dream, and a branch cannot be
refined past its recorded depth. Replaying every candidate over every frozen
trace:

```
              policy  mean cov  worst cov  narrowed   cut off  actions left
--------------------------------------------------------------------------
                pi_1     0.983      0.900       2/9       4/9            24
  adaptive_portfolio     0.983      0.900       2/9       4/9            24
       uct_portfolio     0.985      0.917       2/9       2/9            12
```

While a replay is running it sees 98% of the actions the live grid would have
offered — but **4 of 9 replays of `π₁` ended because the trace ran out, not
because the policy chose to stop**, with 24 live actions still open. That is the
restriction a coverage average cannot see: a replay dying of exhaustion takes no
decision round at the moment it dies. Off-policy evaluation on a frozen history
is exact about what it shows and silent about what it cannot.

### …and against a model that cannot repeat itself

The mock discovery agent is seeded, so the result above proves less than it
looks: of course a recording of a deterministic process reproduces it. The claim
only earns its keep when the environment that produced the history *cannot* be
re-run — which is the only situation in which anyone would want a replay
simulator in the first place.

So: the same policy, the same grid, the same task, deployed live twice through
**Claude Haiku 4.5** — 2 branches × 1 refinement on `circle_packing` at
`n = 26`, baseline `2.1667`. This one needs model access, so it is a script
rather than a subcommand, and it exits non-zero if any replay is inexact:

```bash
python3 scripts/real_replay_fidelity.py --config configs/circle_packing_claude.yaml
```

```
                       b0a0     b0a1      b1a0      b1a1      best
live run 1 (252s)      fail     fail    2.438966  2.497535  2.497535
live run 2 (324s)      fail     fail      fail    2.504756  2.504756
                                       identical across the two runs: False
```

Same code, same policy, different discoveries. Now freeze each of those trees,
write it to disk, load it back, and replay the same policy over it:

```
run 1: exact -- rounds 2/2, probes 4/4, best 2.497535 == 2.497535, 0 score mismatches
run 2: exact -- rounds 2/2, probes 4/4, best 2.504756 == 2.504756, 0 score mismatches
```

Both exact, and exact about the failures too — a `runtime` error and a
`correctness` violation replay as those same failures rather than as missing
cells, which matters because a policy that prunes on failure is reading them.
**The environment is not reproducible; the history it leaves behind is.** That is
precisely the property the method's economy needs: you cannot re-run the
expensive thing, but you can replay it perfectly, at no cost, as often as you
like.

### What a small model actually achieves on the four tasks

One round each, 3 branches × 2 refinements (9 real calls; `lasso` 2 × 1, 4 calls),
Claude Haiku 4.5. **Read this as evidence about the harness, not about the
paper**: the paper runs 1k generations per math task with Gemini-3.1-Pro /
3.7-Flash, so at 9 calls with a small fast model this is a ~100× smaller
experiment answering a different question — does the machinery work end to end on
real tasks, and does the failure taxonomy carry real information?

| task | baseline | best found | verdict |
|---|---|---|---|
| `circle_packing` | `2.1667` | **`2.5868`** | **+19%** — SLSQP over centres and radii jointly |
| `sum_difference` | `1.0` | `1.0` | tie, never beaten |
| `autocorrelation` | `−2.0` | `−2.0` | tie, never beaten |
| `lasso` | `19.956` | `10.239` | `0.51×` — pure-Python route only |

The baselines are programs, not constants: `prepare()` writes `baseline/method.py`
and scores it with the same evaluator as every candidate. They are also not all
equally weak, which is what makes "tie" mean different things above. The circle
packing baseline is a 6×6 grid of equal circles — easy to beat. The
`sum_difference` baseline is the arithmetic progression `[0..11]`, and an AP has
`|A+A| = |A-A| = 23`, so `Γ` is *exactly* `1.0`; beating it requires an MSTD set,
which is genuinely hard. Every successful candidate came back symmetric:

```
  b0a0  |A|=12  |A+A|= 23  |A-A|= 23   MSTD=False     (the baseline set, rediscovered)
  b1a0  |A|=40  |A+A|= 79  |A-A|= 79   MSTD=False
  b2a0  |A|=91  |A+A|=359  |A-A|=361   MSTD=False     Γ = 0.996
  b2a2  |A|=50  |A+A|= 99  |A-A|= 99   MSTD=False
```

One of those is worth dwelling on. `b1a2`'s proposal is titled *"Asymmetric
Pair-Cluster Construction"* and correctly cites its sibling's measured score —
so cross-branch history reading works — yet it emitted the 12-element arithmetic
progression. The proposal and the program disagree, which is exactly why Listing
1 says **"Trust the measured result over what the proposal claims about itself"**
and why the trace records the score rather than the claim.

The `lasso` round is the one where the agent surprised us. Given only the problem
statement it found the native route on its own:

```
b0a0  0.904753  ok             naive pure-Python path
b1a0  10.238503 ok             warm-started coordinate descent (within 2x of sklearn)
b0a1  --        compile_other  NameError: njit          -- numba is not installed
b1a1  --        compile_other  Eigen headers not found  -- it wrote CPP_CODE + -O3 -march=native
```

`b1a1` escalated from Python to C++ unprompted, and the harness classified the
result as `compile_other` — the class the prompt calls "normally repairable",
which is the correct reading: the idea was sound, the environment lacked a
header.

### A fuller world

A fuller round on `circle_packing` — 3 branches × 2 refinements, **9 real
discovery-agent calls, 558s** — produces a world worth replaying:

```
  b0a0  2.586775  ok           b0a1  --  timeout      b0a2  2.499615  ok
  b1a0  --  correctness        b1a1  --  timeout      b1a2  --  timeout
  b2a0  2.560934  ok           b2a1  --  timeout      b2a2  --  timeout
```

Best `2.5868` against the `2.1667` baseline — **+19%**, found by Haiku replacing
the grid packing with SLSQP over centres and radii jointly. Replaying the shipped
policies over that frozen world costs nothing:

```
              policy  probes       best
                pi_1       9     2.5868
  adaptive_portfolio       8     2.5868
       uct_portfolio       8     2.5868
```

Same ceiling, one probe cheaper. Read that as a demonstration that the machinery
works on a real world, not as a measured saving — on a nine-cell grid there is
almost nothing to save, and the seed-level evidence for the saving is
[further down](#across-seeds-the-saving-is-real-the-quality-tie-is-not).

The failure mix is worth more than the scores. That round ran with a 60-second
candidate budget, and every depth-1 refinement blew it: asked to improve a
packing optimizer, the agent reached for more restarts and more iterations, and
the evaluator stopped them before it could judge them. Nothing was wrong with the
proposals — the budget was measuring the evaluator's patience. `b0a2` then
recovered to a valid `2.4996` from a parent that had timed out, which is the
failure taxonomy working as intended. The shipped config now allows 240s, so
rerunning it will not reproduce this table.

### Dreaming's replay scores are optimistic, but its *ranking* holds

```bash
dreamrsi generalize --config configs/synthetic.yaml --holdout-last 1
```

`H_t` is split temporally — dream on the earlier worlds, score on the later ones,
the split the loop itself faces — and every candidate is scored on both sides.
The candidates are dreaming's own: `v0` is the incumbent `π₁`, `v1` and `v2` the
two library policies the mock developer proposes.

```
 version   pool reward  holdout reward        gap
--------------------------------------------------
       0      0.573317        0.489866  +0.083451
       1      0.591895        0.513927  +0.077968
       2      0.581720        0.510073  +0.071647
--------------------------------------------------
selected on pool: v1   best on holdout: v1
selection regret: 0.000000   mean gap: +0.077689   rank corr: +1.000
non-regression transfers to held-out worlds: True
```

Replay reward is **systematically optimistic** — every candidate scores ~0.078
higher on the worlds it was selected on than on the world it was not (holding out
two worlds instead of one widens the gap to +0.282). But the gap is roughly
*uniform across candidates*, and selection depends only on their order: rank
correlation is `+1.000` and selection regret is `0`. An oracle with access to the
held-out world would have picked the same version.

This is the distinction that matters for reading any offline number here. The
absolute replay rewards should not be believed as predictions of live
performance; the *comparison between candidates* is what dreaming needs, and on
this task it survives.

### Across seeds: the saving is real, the quality tie is not

```bash
dreamrsi benchmark --config configs/synthetic.yaml --seeds 0 1 2 3 4 5 6 7
```

The Section 4 comparison as a **paired** experiment — for a given seed both arms
see the same task, agent and round-1 policy, so the within-pair difference
isolates dreaming:

```
 seed     fixed best     dream best  dream @budget  fixed calls  dream calls   ratio
-------------------------------------------------------------------------------------
    0     -15.199224     -15.199224      -7.014438          468          297   1.576
    1     -12.226702     -18.784452      -3.776649          468          252   1.857
    2      -6.761106      -6.761106      -6.761106          468          468   1.000
    3     -11.521382     -15.690958     -15.690958          468          285   1.642
    4      -5.333137     -12.265277     -12.265277          468          226   2.071
    5     -13.576364     -22.280591     -22.280591          468          235   1.991
    6     -18.092499     -18.092499      -6.923388          468          250   1.872
    7     -13.136500     -13.136500     -13.136500          468          238   1.966
-------------------------------------------------------------------------------------
quality @equal rounds: mean -3.295462  95% CI [-5.816839, -1.042394]  W/T/L 0/4/4
quality @equal calls:  mean +0.999751  95% CI [-3.928387, +5.929276]  W/T/L 3/2/3
calls (fixed / dream): mean 1.747x     95% CI [1.501, 1.946]  cheaper on 7/8
```

Two readings, and only the second is meaningful:

* **At equal rounds** Dream-RSI loses on quality — `0/4/4`, and the interval
  excludes zero. But an improved exploration policy stops earlier, so after the
  same number of rounds the two arms have done *different amounts of work*.
  Spending less is not the same as exploring better, and a lower score here says
  almost nothing.
* **At equal calls** — the dreaming arm run on until it has spent the fixed arm's
  full 468-call budget — the difference disappears: `3/2/3`, with an interval
  comfortably straddling zero.

So on this offline stand-in the honest summary is: **dreaming reliably cuts
discovery-agent calls (1.75×, 7/8 seeds) and, at matched budget, neither improves
nor harms discovered quality.** The cost half of the paper's claim reproduces;
the quality half is not detectable at this scale. The `compare` table in the
Quickstart is the seed-0 slice of the first row above.

No significance test is applied. With a mock agent on a synthetic task the
sampling distribution is a property of this repository's stand-ins, not of
Dream-RSI, so a p-value would be precision about the wrong population. The
intervals show spread, not a certified effect.

### The worlds have to evolve

The paper's title claims two things, and the second is testable on its own.
`--ablate` pins the simulator pool to a sliding window with `pool_window`, so the
offline phase never accumulates evidence — it re-runs the whole comparison
dreaming over only the newest world:

```bash
dreamrsi benchmark --config configs/synthetic.yaml --ablate
```

```
        pool  W/T/L @rounds  W/T/L @budget  mean call ratio  cheaper seeds
----------------------------------------------------------------------------
  all of H_t          0/4/4          3/2/3            1.747            7/8
      last 1          0/3/5          0/3/5            1.874            8/8
```

Restricted to one world, dreaming **never once beats** fixed exploration at
matched budget and loses on 5 of 8 seeds — while cutting calls slightly *harder*
(1.87×, 8/8). That combination is the signature of a policy that has learned to
stop early rather than to explore well: it is selecting against a single frozen
trace and carrying the result into a world that does not resemble it. The growing
pool is what keeps that honest, so the advantage really does come from the
history *accumulating*, not merely from replaying one.

---

## Writing a policy

A policy is a plain Python module defining `NAME` and
`class OptimalPolicy(LLMDesignedMethod)`. It is imported through
[`policy/loader.py`](dreamrsi/policy/loader.py), which first runs a static check
for the constraints the prompt states — the class and `NAME` exist, `solve` and
`plan_grid` are implemented, `plan_grid` never returns `None`, `plan_grid` does not
merely delegate to `context.bootstrap_plan(...)` on every path, and `solve` does
not read `question.best_so_far` / `question.budget_spent` (bookkeeping only) or
reach into unrevealed outcomes.

```python
from see.policy.api import (
    LLMDesignedMethod, SimResult, _budget_done, _record_curve, finalize_result,
)

NAME = "OptimalPolicy"


class OptimalPolicy(LLMDesignedMethod):
    def __init__(self, config=None):
        super().__init__(config)
        self.beta = float(self.config.get("beta", 0.6))     # exactly one scalar knob

    def solve(self, question, budget=None):
        question.reset()
        res, closed = SimResult(), set()
        while not _budget_done(question, budget):
            prefix = question.observed()
            update_closed(closed, prefix, question)
            batch = select_batch(prefix, question, closed)
            if not batch:
                break
            question.probe_batch(batch, on_reveal=lambda _: _record_curve(res, question))
        return finalize_result(question, res)

    def plan_grid(self, context):                            # runs before the next live grid
        return GridPlan(branch_count=..., refine_count=..., reason="...")
```

`see.policy.api` is a shim over `dreamrsi.policy.api` so that policies written
against the paper's prompt import verbatim.

Three policies ship:

| File | Mechanism |
|---|---|
| [`policy/initial_policy.py`](dreamrsi/policy/initial_policy.py) | `π₁` — the manual **parallel refining** strategy: `W` independent workspaces, each refining its own candidate. Also the Recursive Fixed Exploration baseline. |
| [`policy/library/adaptive_portfolio.py`](dreamrsi/policy/library/adaptive_portfolio.py) | Trajectory-ranked dynamic portfolio: exploitation + exploration + at most one recovery per batch, cumulative-evidence closure, portfolio-level stagnation stop. |
| [`policy/library/uct_portfolio.py`](dreamrsi/policy/library/uct_portfolio.py) | Upper-confidence allocation: an optimism bonus decaying in attempts, so a weak *shallow* score cannot bury a direction before it has been given depth. |

The library doubles as the mock policy developer's proposal set, which is what
lets the recursive loop run with no model access.

### `beta` has three distinct roles

Per the prompt, and implemented that way:

1. **Within one episode** it is *fixed*, and every behavioural threshold is
   routed through a single `_schedule(beta) -> dict`. High beta means more width,
   deeper patience, weaker pruning; low beta means fewer probes, earlier
   stagnation stops, stronger pruning.
2. **During offline evaluation** a fixed beta grid is swept, to measure whether
   the policy exposes a real attainment/work/parallelism trade-off. This is not
   online beta adaptation.
3. **When proposing the next version** a default is baked in once, using the
   cross-cycle rule over the last 2–3 live manifests and their archived sweeps.

---

## Agents

Both agents are pluggable and neither is touched by the improvement loop —
Dream-RSI updates only the exploration-policy code.

| Kind | Discovery agent | Policy developer |
|---|---|---|
| `mock` | seeded hill-climber; reads completed rounds for its root directions | picks from the policy library, rebakes the default beta |
| `cli` | runs a coding-agent CLI (Gemini CLI, Claude Code, Codex…) in the attempt directory with Listing 1 | runs the same CLI on `method.py` with Listing 2 |
| `llm` | one Anthropic/OpenAI call per `CONTINUE` | one call per revision |

The paper drives Gemini-3.1-Pro and Gemini-3.7-Flash through the Gemini CLI; the
`cli` agent takes any command template:

```yaml
discovery_agent: cli
policy_developer: cli          # inherits `agent` below unless `policy_agent` is set
task_config:
  agent:
    command: "gemini --yolo"
    timeout: 900               # seconds one discovery-agent call may take
```

The template substitutes `{prompt_file}`, `{node_dir}`, `{run_dir}`,
`{history_dir}`, `{baseline_dir}`, `{problem_file}` and `{program}`; the prompt
is *also* piped on stdin, which is what most CLIs accept by default. Only
`{history_dir}` is supplied to both agents, so it is the safe placeholder for a
shared template — anything else raises an error naming the placeholder rather
than silently dropping the argument.

Those paths exist because Listing 1 tells the agent to read the whole history
before proposing, and sibling attempts, the baseline and `problem.md` all sit
*above* the attempt directory the CLI is launched in. A CLI that only trusts its
working directory has to be handed them explicitly.

### Verified: Claude Code on `circle_packing`

[`configs/circle_packing_claude.yaml`](configs/circle_packing_claude.yaml) is a
worked, tested example — pure Python, no compiler, a cheap evaluator, and a grid
small enough that the whole round is a handful of model calls:

```yaml
task_config:
  agent:
    command: >-
      claude -p --model claude-haiku-4-5 --permission-mode acceptEdits
      --add-dir {history_dir}
      --allowedTools Write Edit Read Glob Grep
```

```bash
dreamrsi run --config configs/circle_packing_claude.yaml
```

Three details are load-bearing. `--permission-mode acceptEdits` is what keeps
the CLI non-interactive without `--dangerously-skip-permissions`, which refuses
to run as root. `--add-dir {history_dir}` is what lets the agent actually open
the history the prompt tells it to read. And `--allowedTools` is variadic, so it
has to come last — the prompt reaches the CLI on stdin, not as an argument.

**The prompts are reproduced verbatim** from Appendix B — Listing 1 in
[`prompts/exploration_prompt.md`](dreamrsi/prompts/exploration_prompt.md) and
Listing 2 in
[`prompts/policy_improvement_prompt.md`](dreamrsi/prompts/policy_improvement_prompt.md)
— with their `$node_dir` and `{method_file}` variables instantiated by the
runtime, as the paper describes. `dreamrsi prompt exploration` prints them.

---

## Tasks

| Task | Paper | Status |
|---|---|---|
| `lasso` | Appendix A Problem 1 — Lasso regularization path | full harness: 17 synthetic regimes, disjoint correctness/timing splits, `F_k(w) ≤ F_k(w_sklearn) + 1e-6`, `R_search = (Π tᵢ)^(-1/\|I\|)` |
| `sum_difference` | Problem 2 — maximise `Γ(A)` | full |
| `circle_packing` | Problem 3 — maximise `Σ rᵢ`, `n ∈ {26, 32}` | full |
| `autocorrelation` | Problem 4 — all three inequalities | full, on the piecewise-constant relaxation |
| `kernelbench` | Section 4.3 — VGG16, LayerNorm, ConvDiv, ConvMax | adapter only; shells out to your own KernelBench command on a CUDA box |
| `synthetic` | — | offline stand-in used by the tests and the demo |

A candidate for `lasso` may be **native** (`CPP_CODE` + `COMPILE_FLAGS`, driven
over a binary stdin/stdout protocol) or **pure Python** (`solve_path(X, y,
lambdas)`). The baseline is scikit-learn's coordinate descent.

### The solver Dream-RSI discovered

[`discovered/lasso_path/method.py`](discovered/lasso_path/method.py) is the
complete 847-line program from Appendix C: strong-rule screening with adaptive
Cauchy–Schwarz KKT pruning, disjoint active-set bookkeeping, lazy Gram-matrix
construction and hardware-aware layout. It is a normal candidate for the `lasso`
task:

```bash
python3 scripts/verify_discovered_lasso.py        # needs g++ and Eigen
```

It compiles and passes the benchmark's objective check on all six held-out
instances of the default 9-instance grid — which is the evidence that the
847-line transcription is faithful.

It does **not** pass the full 17-instance grid: on a correlated `p >> n` instance
it exceeds the `1e-6` margin by `2.1e-6`. Nor is it fast here — 13× slower than
scikit-learn on the default timing instances, on a box that had both OpenMP and
`-march=native`. Neither result says anything about the published solver on the
published benchmarks, and both have identifiable causes in *this* harness (the
timing split excludes the `p >> n` regimes the solver is built for; the
correctness reference is itself un-converged by about the size of the margin it
defines). [`discovered/lasso_path/README.md`](discovered/lasso_path/README.md)
works through both, and why the timings here are **not** comparable to
Table 1(a).

---

## Run layout

Directory names follow the paths the prompts reference, so an agent written
against Appendix B can be pointed straight at a run:

```
runs/<name>/
  history/
    baseline/                        parallel-refine floor, with eval/score.json
    r0000_init/method.py             π₁
    r0001_dream/
      attempts/attempt_b0a3/         proposal.md, method.*, eval/score.json, error.txt
      trace.json                     the frozen discovery tree
      live_cycle_manifest.json       prefix-safe sidecar plan_grid may read
      policy_used.py
      proposal_results/
        beta_sweep.json              every version's pareto summary and per-beta frontier
        policy_execution_traces.jsonl  one replay episode per (frozen trace, beta)
        v000_method.py … v003_method.py
  trace_pool/
    iter1/live_cycle_manifest.json
    _current/live_cycle_manifest.json
  report.json
```

`policy_execution_traces.jsonl` records, per decision round, the prefix size,
the selected batch and the revealed outcomes — the between-round feedback the
improvement prompt tells the agent to diagnose from, and never to read inside
`solve()`.

The [Evidence](#evidence) commands write outside that tree, because their
artifacts are measurements of a run rather than inputs to one:

```
runs/<name>-fidelity/fidelity.json       per-probe trajectory diff + support audit
runs/<name>-generalization.json          per-version pool vs held-out rewards
runs/<name>-benchmark/benchmark.json     per-seed rows and the aggregate summary
```

---

## CLI

```bash
dreamrsi run     --config configs/synthetic.yaml       # one recursive loop
dreamrsi compare --config configs/synthetic.yaml       # Dream-RSI vs fixed
dreamrsi eval    --policy P.py --pool runs/x/history   # beta sweep over frozen traces
dreamrsi dream   --policy P.py --pool runs/x/history   # offline improvement only
dreamrsi inspect runs/x/history/r0001_dream/trace.json -v
dreamrsi tasks | dreamrsi prompt exploration | dreamrsi prompt policy
```

The [Evidence](#evidence) commands:

```bash
dreamrsi fidelity   --config configs/synthetic.yaml            # is replay exact?
dreamrsi generalize --config configs/synthetic.yaml --holdout-last 1
dreamrsi generalize --pool runs/x/history --holdout-last 2     # audit your own run
dreamrsi benchmark  --config configs/synthetic.yaml --seeds 0 1 2 3
dreamrsi benchmark  --config configs/synthetic.yaml --ablate   # + evolving-worlds ablation
```

`fidelity` exits non-zero if any replay is inexact, so it works as a regression
gate. `benchmark` gives the dreaming arm `3 × --rounds` rounds so it can spend the
fixed arm's full call budget; `--equal-budget-rounds` overrides that.

Progress goes to stderr; stdout stays machine-readable.

---

## Faithfulness

**Reproduced from the paper.** The three-stage loop and its recursive structure;
the tree/grid decision interface and its legality rules; batch cost accounting;
Equation (1); the `pareto.reward` objective and the beta sweep; the
non-regression property of policy selection; `plan_grid` and its prefix-safe
context; the success semantics and failure taxonomy the prompt insists on; the
two prompts verbatim; the discovered Lasso solver; the task definitions of
Appendix A.

**Filled in, because the paper does not specify it.** Exact cell-id scheme;
`attainment` normalised against the trace ceiling; `pareto.auc` as the envelope
area over the sweep; the default `β₁`, `β₂`, `λ` and beta grid; the manifest
schema; the static policy checks; the mock agents; the synthetic task; the
scikit-learn reference and the synthetic instance grid for `lasso`; **the
baseline program of every task** — the grid packing, the arithmetic progression,
the uniform density. These are reasonable readings, not the authors' choices, and
anything that depends on them (absolute reward values, say) will not match the
original.

The baselines deserve singling out, because "beat the baseline" is only as
meaningful as the baseline. They are not uniformly weak: the circle-packing grid
is easy to beat, while the `sum_difference` arithmetic progression scores exactly
`Γ = 1.0` and beating it requires constructing an MSTD set. A tie against one is
not a tie against the other.

**Measured, not assumed.** Three claims the method depends on are checked rather
than taken on trust, and they do not all come out the same way — see
[Evidence](#evidence). A frozen history replays **exactly** (9/9 through
serialisation with the mock agent, and again on two live Claude Haiku 4.5 runs
that did not reproduce each other), so the simulator does not lie; but it is
exact only about the region that was explored, and 4 of 9 off-policy replays
ended because the trace ran out rather than because the policy chose to stop.
Replay rewards are
systematically **optimistic** about held-out worlds (+0.078 at one held-out
world, +0.282 at two) while preserving the candidate *ranking* that selection
actually uses, so selection regret is zero here. Across eight seeds the call
saving reproduces (1.75×, cheaper on 7/8) and the quality difference is
undetectable once the two arms are given the same budget — but pinning the
simulator pool to a single world removes even that, which is direct evidence for
the *evolving* half of "evolving worlds".

**Not reproduced.** The paper's headline numbers. Table 1(a) needs
Gemini-3.1-Pro / Gemini-3.7-Flash through the Gemini CLI, the SimpleTES
instance suite and six held-out downstream datasets; Table 1 (math) needs 1k
generations per task; Figure 4 needs KernelBench on GPUs. None of that runs in
this repository without your own model access and hardware. The numbers above
come from the offline synthetic task and show only that the *mechanism* behaves
as described — dreaming selects a policy that reaches comparable quality with
fewer discovery-agent calls. They are evidence about this reimplementation and
its mock agents, not about Dream-RSI's performance on the paper's benchmarks.

---

## Tests

```bash
python3 -m pytest -q          # 121 passed, 1 skipped
EIGEN_INCLUDE=/path/to/eigen3 python3 -m pytest -q   # 122 passed -- nothing skipped
```

They cover tree invariants, world legality and cost accounting, both objectives,
the policy API and its static checks, every shipped policy at three betas, the
task evaluators' scoring and failure classification, the extracted solver, the
prompts, the CLI, and the full loop including the non-regression guarantee.
[`tests/test_experiments.py`](tests/test_experiments.py) covers the evidence
layer: replay exactness through serialisation (and that a tampered trace or a
diverging trajectory is *caught* rather than averaged away), support coverage on
complete and partial traces, the temporal split and its regret arithmetic, and
budget matching. The single skip is the discovered solver's compile-and-run
test, which needs Eigen:

```bash
EIGEN_INCLUDE=/path/to/eigen3 python3 -m pytest -q
```

---

## Citation

Please cite the original paper. `CITATION.cff` in this repository carries both
entries.

```bibtex
@article{zheng2026dreamrsi,
  title  = {Dream-RSI: Recursive Self-Improvement through Evolving Worlds},
  author = {Zheng, Tong and Wu, Xidong and Zhang, Zheng and He, Zhankui and
            Zhang, Chaoyi and Coleman, Benjamin and Wei, Ruoqiao and Bai, Di and
            Liu, Haolin and Liu, Rui and Wang, Xue and Zhuan, Yue and
            Kang, Wang-Cheng and Xiang, Renkai and Huang, Heng and
            Cheng, Xinwu and Guo, Yunsong},
  year   = {2026},
  url    = {https://www.dream-rsi.com}
}
```

## License

Apache-2.0. Independent reproduction, not affiliated with or endorsed by the
paper's authors. The prompts under `dreamrsi/prompts/` and the solver under
`discovered/lasso_path/` are reproduced from the paper's appendices and remain
the authors' work.
