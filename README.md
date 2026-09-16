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
the incumbent on the accumulated history. These are numbers from the offline
synthetic task, not a reproduction of the paper's benchmarks — see
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
policy_developer: cli
task_config:
  agent:
    command: "gemini --yolo"      # {prompt_file} {node_dir} {program} are substituted
```

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

It compiles and passes the benchmark's objective check on every held-out
instance. See [`discovered/lasso_path/README.md`](discovered/lasso_path/README.md)
for how the listing was transcribed and why the timings here are **not**
comparable to Table 1(a).

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
scikit-learn reference and the synthetic instance grid for `lasso`. These are
reasonable readings, not the authors' choices, and anything that depends on them
(absolute reward values, say) will not match the original.

**Not reproduced.** The paper's headline numbers. Table 1(a) needs
Gemini-3.1-Pro / Gemini-3.7-Flash through the Gemini CLI, the SimpleTES
instance suite and six held-out downstream datasets; Table 1 (math) needs 1k
generations per task; Figure 4 needs KernelBench on GPUs. None of that runs in
this repository without your own model access and hardware. The demo numbers
above come from the offline synthetic task and show only that the *mechanism*
behaves as described: dreaming selects a better policy on the accumulated
history, and redeploying it reaches the same quality with fewer
discovery-agent calls.

---

## Tests

```bash
python3 -m pytest -q          # 97 passed, 1 skipped
```

They cover tree invariants, world legality and cost accounting, both objectives,
the policy API and its static checks, every shipped policy at three betas, the
task evaluators' scoring and failure classification, the extracted solver, the
prompts, the CLI, and the full loop including the non-regression guarantee. The
single skip is the discovered solver's compile-and-run test, which needs Eigen:

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
