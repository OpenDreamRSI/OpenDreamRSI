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
1(a) — in particular, a single-threaded build on a machine without libomp gives
up most of what the program is designed to exploit.
