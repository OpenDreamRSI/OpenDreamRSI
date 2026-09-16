# Lasso regularization-path solver discovered by Dream-RSI

`method.py` is the complete solver from **Appendix C, Listing 3** of the paper,
transcribed from the published PDF. It is 847 lines: a Python module holding one
`EVOLVE-BLOCK` with a `CPP_CODE` string and `COMPILE_FLAGS = ["-fopenmp",
"-ffast-math"]`.

## What it does

Section 4.1 describes it as follows. Unlike SimpleTES, which switches between
LARS and coordinate descent according to problem dimensions, this solver
introduces adaptivity *inside* the active-set optimisation:

* **strong-rule screening** with **Cauchy–Schwarz KKT pruning**, recomputing
  exact gradients only when the bound cannot certify a feature, and falling back
  to a full refresh when pruning stops being effective (`use_cs`, `to_compute`,
  `lambdas_since_reset`);
* **disjoint-partition active-set bookkeeping** — `screened_list` /
  `unscreened_list` with O(1) swap-deletion so a feature is only ever in one
  partition;
* **lazy Gram-matrix construction**, grown by doubling with 64-byte aligned
  reallocation and a SIMD 4x register-blocked column precomputation;
* **hardware-aware layout** — column padding to a multiple of 8 rows, a fused
  transposition-precomputation-padding pass over `X`, `#pragma omp simd`
  reductions and `ASSUME_ALIGNED` hints throughout.

## Running it

It is a normal candidate for the `lasso` task, so:

```bash
python3 scripts/verify_discovered_lasso.py
```

compiles it, checks `F_k(w_k) <= F_k(w_k_sklearn) + 1e-6` on held-out instances,
then times the complete path on the timing instances.

Requirements: a C++ compiler and **Eigen** (`brew install eigen`,
`apt install libeigen3-dev`, or `--eigen /path/to/eigen3`). Two portability
fallbacks are applied automatically by the harness and change throughput only,
never results:

* on a toolchain without OpenMP (Apple clang, say), `-fopenmp` is dropped and
  `<omp.h>` is satisfied by `dreamrsi/tasks/_ompshim/omp.h`, so the program
  builds and runs single-threaded;
* `-march=native` is dropped if the compiler rejects it.

## Fidelity of the transcription

The listing was extracted from the PDF's text layer (not OCR) with per-glyph
column positions, so indentation and wrapped lines are reconstructed rather than
guessed. Checks that the result is the intended program:

* exactly 847 lines, matching the listing's own line numbering;
* balanced `{}`, `()` and `[]`;
* the module parses as Python and `CPP_CODE` compiles cleanly with `g++
  -std=c++17` against Eigen;
* the compiled solver passes the benchmark's objective-value check on every
  correctness instance.

`tests/test_tasks.py` runs the first three unconditionally and the last one
whenever Eigen is present.

## What the timings here do and do not show

The paper reports this solver beating `sklearn` and `glmnet` on six held-out
downstream datasets (Gisette, RCV1, DNA, Leukemia, Colon, Duke Breast) on the
authors' hardware, with OpenMP. This repository ships synthetic instances and
its own timing harness, so the numbers you get are **not** comparable to Table
1(a).

A measured example, on a 4-core Linux box **with** OpenMP and `-march=native`
both available — so the usual "it built single-threaded" excuse does not apply:

```
$ python3 scripts/verify_discovered_lasso.py          # the defaults: 9 instances, 3 timing
sklearn baseline   R_search = 8.247
discovered solver  R_search = 0.630                   # 13x slower
objective check passed on 6/6 held-out instances
```

The correctness result is the meaningful one: the transcription is faithful. The
timing is an artefact of **which instances get timed**. `timing_names` takes the
*last* `n_timing` entries of `INSTANCE_GRID`, and this script defaults to
`--instances 9`, so the timed instances are `(1000, 200)`, `(1000, 500)` and
`(1000, 1000)` — every one of them `p <= n`. The `p >> n` regimes in the grid
(`(200, 500)`, `(500, 1000)`, `(300, 2000)`) are cut off by the `[:9]` slice.
Strong-rule screening and a cached Gram matrix are precisely the machinery that
pays off when `p >> n` and screening can discard most features; on dense
`p <= n` instances there is little to screen and the Gram construction is
overhead. So the default invocation times this solver in the regime it was
*least* designed for.

Running the full grid instead (`--instances 17 --timing 5`, matching
`configs/lasso.yaml`) does not produce a better timing either, because it stops
earlier:

```
FAILED (correctness):
n200_p500_s10_c0.5_seed12354: objective check failed at k=40:
  F_k(w) = 0.272044540 > 0.272042398 + 1e-06
```

That is a `2.1e-06` excess against a `1e-06` margin, on a *correlated* `p >> n`
instance (`corr = 0.5`). Two things are true about it and both matter:

* the solver really is slightly loose here — against a properly converged
  solution it sits about `3.1e-06` above the optimum;
* **the yardstick is not converged either.** `lasso_path` builds the reference
  with scikit-learn's default tolerance and emits `ConvergenceWarning` on these
  instances; re-solving `k=40` to `tol=1e-14` shows the reference is itself
  `9.8e-07` above the true optimum — the same order as the acceptance margin it
  defines. An absolute `1e-06` bar measured against a reference carrying `~1e-06`
  of its own slack is not a sharp test.

The correlated regimes are this repository's invention, not the paper's (see
[Faithfulness](../../README.md#faithfulness)), so none of this is evidence about
the published solver on the published benchmarks. It is a caution about reading
any number this harness produces for it.
