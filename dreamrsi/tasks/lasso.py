"""Problem 1 of Appendix A -- the Lasso regularization path.

Given ``X in R^{n x p}``, ``y in R^n`` and a decreasing sequence
``lambda_1 > ... > lambda_K``::

    F_k(w) = (1 / 2n) ||y - X w||_2^2 + lambda_k ||w||_1

A candidate solver returns approximate coefficients ``W = (w_1, ..., w_K)``.  It
passes the benchmark's objective-value check if

    F_k(w_k) <= F_k(w_k_sklearn) + 1e-6   for every k,

with correctness checked on fresh instances distinct from the timing instances.
If any required check fails the search score is zero.  Otherwise, with ``I`` the
timing instances and ``t_i`` the time to compute the complete path on instance
``i``, the search score is

    R_search = (prod_{i in I} t_i)^(-1 / |I|)

i.e. the inverse geometric mean of the path times -- larger is better.

A candidate ``method.py`` may take either route:

* **native** -- define ``CPP_CODE`` (and optionally ``COMPILE_FLAGS``).  The
  source is compiled once and then driven over stdin/stdout with the binary
  protocol below.  This is the shape of the solver Dream-RSI discovered, which
  ships in ``discovered/lasso_path/method.py``.
* **python** -- define ``solve_path(X, y, lambdas) -> W`` of shape ``(K, p)``.

Binary protocol (little-endian, native sizes)::

    stdin :  int32 n, int32 p, int32 K,
             float64 X[n*p] (row-major), float64 y[n], float64 lambdas[K]
    stdout:  float64 coef_path[p*K]  (column-major p x K, i.e. w_k[j] at k*p + j)
"""

from __future__ import annotations

import os
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .base import EvalResult, Task
from . import register

HERE = os.path.dirname(os.path.abspath(__file__))
OMP_SHIM_DIR = os.path.join(HERE, "_ompshim")

#: Where Eigen usually lives.  ``eigen_include`` in the task config overrides.
EIGEN_CANDIDATES = (
    "/opt/homebrew/include/eigen3",
    "/usr/local/include/eigen3",
    "/usr/include/eigen3",
    "/opt/homebrew/opt/eigen/include/eigen3",
)

BASELINE = '''"""Baseline Lasso path solver: scikit-learn's coordinate descent."""

import numpy as np
from sklearn.linear_model import lasso_path


def solve_path(X, y, lambdas):
    """Return W of shape (len(lambdas), p)."""
    _, coefs, _ = lasso_path(X, y, alphas=np.asarray(lambdas, dtype=float))
    return np.asarray(coefs).T
'''


# -- instances ---------------------------------------------------------------
def make_instance(seed: int, n: int, p: int, sparsity: int, corr: float,
                  n_lambda: int = 50) -> Dict[str, Any]:
    """One synthetic regression instance plus its lambda path."""
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((n, p))
    if corr > 0:
        common = rng.standard_normal((n, 1))
        base = np.sqrt(1.0 - corr) * base + np.sqrt(corr) * common
    X = np.ascontiguousarray(base, dtype=np.float64)
    w = np.zeros(p)
    idx = rng.choice(p, size=min(sparsity, p), replace=False)
    w[idx] = rng.standard_normal(len(idx)) * 2.0
    y = X @ w + rng.standard_normal(n) * 0.5
    y = np.ascontiguousarray(y, dtype=np.float64)

    lam_max = float(np.max(np.abs(X.T @ y)) / n)
    lambdas = np.ascontiguousarray(
        lam_max * np.logspace(0, -3, n_lambda, base=10.0), dtype=np.float64)
    return {"name": f"n{n}_p{p}_s{sparsity}_c{corr:g}_seed{seed}",
            "X": X, "y": y, "lambdas": lambdas}


#: Seventeen instances covering the regimes SimpleTES sweeps: dimensionality,
#: sparsity, feature correlation and active-set structure.
INSTANCE_GRID: Tuple[Tuple[int, int, int, float], ...] = (
    (200, 50, 5, 0.0), (200, 200, 10, 0.0), (200, 500, 10, 0.0),
    (500, 100, 10, 0.0), (500, 500, 20, 0.0), (500, 1000, 20, 0.0),
    (1000, 200, 20, 0.0), (1000, 500, 30, 0.0), (1000, 1000, 40, 0.0),
    (200, 500, 10, 0.5), (500, 500, 20, 0.5), (1000, 1000, 40, 0.5),
    (300, 2000, 15, 0.0), (300, 2000, 15, 0.7),
    (2000, 100, 10, 0.0), (2000, 300, 25, 0.3), (800, 800, 5, 0.9),
)


def objective(X: np.ndarray, y: np.ndarray, w: np.ndarray, lam: float) -> float:
    n = X.shape[0]
    resid = y - X @ w
    return float(resid @ resid / (2.0 * n) + lam * np.abs(w).sum())


# -- native candidates -------------------------------------------------------
def find_eigen(explicit: Optional[str] = None) -> Optional[str]:
    """Locate Eigen: explicit config, then ``$EIGEN_INCLUDE``, then usual prefixes."""
    env = os.environ.get("EIGEN_INCLUDE")
    for cand in [explicit, env, *EIGEN_CANDIDATES]:
        if cand and os.path.exists(os.path.join(cand, "Eigen", "Dense")):
            return cand
    return None


def has_openmp(compiler: str) -> bool:
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "probe.cpp")
        with open(src, "w", encoding="utf-8") as fh:
            fh.write("#include <omp.h>\nint main(){return omp_get_max_threads()>0?0:1;}\n")
        proc = subprocess.run([compiler, "-fopenmp", src, "-o", os.path.join(d, "probe")],
                              capture_output=True, text=True)
        return proc.returncode == 0


def compile_native(cpp_code: str, flags: Sequence[str], workdir: str, *,
                   compiler: str = "g++", eigen_include: Optional[str] = None,
                   extra: Sequence[str] = ("-O3", "-std=c++17", "-march=native"),
                   ) -> Tuple[Optional[str], str]:
    """Compile a candidate solver; returns ``(binary_path, log)``.

    Two portability fallbacks keep a discovered kernel buildable on toolchains
    it was not tuned on: ``-fopenmp`` is dropped (and ``<omp.h>`` satisfied by a
    bundled shim) when the compiler has no OpenMP, and ``-march=native`` is
    dropped when the compiler rejects it.  Both change throughput, never
    results.
    """
    eigen = find_eigen(eigen_include)
    if eigen is None:
        return None, ("Eigen headers not found. Install Eigen (brew install eigen / "
                      "apt install libeigen3-dev) or set task_config.eigen_include.")
    src = os.path.join(workdir, "solver.cpp")
    with open(src, "w", encoding="utf-8") as fh:
        fh.write(cpp_code)
    out = os.path.join(workdir, "solver")

    use_flags = list(flags)
    includes = ["-I" + eigen]
    if "-fopenmp" in use_flags and not has_openmp(compiler):
        # Keep the program buildable on toolchains without libomp: drop the
        # flag and satisfy #include <omp.h> with the bundled shim.
        use_flags = [f for f in use_flags if f != "-fopenmp"]
        includes.append("-I" + OMP_SHIM_DIR)
    attempts = [list(extra)]
    if "-march=native" in extra:
        attempts.append([f for f in extra if f != "-march=native"])

    log = ""
    for opts in attempts:
        cmd = [compiler, *opts, *use_flags, *includes, src, "-o", out]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        log = " ".join(cmd) + "\n" + (proc.stderr or "")
        if proc.returncode == 0:
            return out, log
    return None, log


def launch_overhead(binary: str, timeout: float = 60.0, samples: int = 3) -> float:
    """Median cost of starting the solver process on a degenerate instance.

    The benchmark times "the time required to compute the complete
    regularization path".  A native candidate is driven as a subprocess, so its
    measured wall time also contains process creation and dynamic linking, which
    an in-process Python candidate never pays.  Measuring that floor once and
    subtracting it keeps the two routes comparable.
    """
    tiny = {"X": np.ones((1, 1)), "y": np.zeros(1), "lambdas": np.array([1.0])}
    times = []
    for _ in range(samples):
        try:
            _, elapsed = run_native(binary, tiny, timeout, overhead=0.0)
        except Exception:  # noqa: BLE001 - a solver may reject a 1x1 instance
            return 0.0
        times.append(elapsed)
    return statistics.median(times) if times else 0.0


def run_native(binary: str, inst: Dict[str, Any], timeout: float,
               overhead: float = 0.0) -> Tuple[np.ndarray, float]:
    X, y, lam = inst["X"], inst["y"], inst["lambdas"]
    n, p = X.shape
    payload = (np.array([n, p, lam.size], dtype=np.int32).tobytes()
               + X.tobytes() + y.tobytes() + lam.tobytes())
    started = time.perf_counter()
    proc = subprocess.run([binary], input=payload, capture_output=True, timeout=timeout)
    elapsed = time.perf_counter() - started
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or b"").decode("utf-8", "replace")[-2000:] or
                           f"solver exited with code {proc.returncode}")
    expected = p * lam.size * 8
    if len(proc.stdout) < expected:
        raise RuntimeError(f"solver wrote {len(proc.stdout)} bytes, expected {expected}")
    coefs = np.frombuffer(proc.stdout[:expected], dtype=np.float64)
    # coef_path is a column-major (p, K) matrix: w_k[j] lives at k * p + j.
    return coefs.reshape(lam.size, p), max(elapsed - overhead, 1e-9)


def run_python_solver(module, inst: Dict[str, Any]) -> Tuple[np.ndarray, float]:
    started = time.perf_counter()
    W = module.solve_path(inst["X"], inst["y"], inst["lambdas"])
    elapsed = time.perf_counter() - started
    return np.asarray(W, dtype=float), elapsed


# -- task --------------------------------------------------------------------
class LassoPathTask(Task):
    """Discover a fast, numerically correct Lasso regularization-path solver."""

    name = "lasso"
    eval_program = "method.py"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        cfg = self.config
        self.n_instances = int(cfg.get("n_instances", 17))
        self.n_timing = int(cfg.get("n_timing", 5))
        self.n_lambda = int(cfg.get("n_lambda", 50))
        self.repeats = int(cfg.get("repeats", 3))
        self.tol = float(cfg.get("tol", 1e-6))
        self.timeout = float(cfg.get("timeout", 300.0))
        self.compiler = cfg.get("compiler", "g++")
        self.eigen_include = cfg.get("eigen_include")
        self.seed = int(cfg.get("seed", 12345))
        # Subtract the subprocess launch floor so native and in-process
        # candidates are timed on the same basis.
        self.subtract_overhead = bool(cfg.get("subtract_overhead", True))
        self._instances: Optional[List[Dict[str, Any]]] = None
        self._reference: Optional[Dict[str, List[float]]] = None

    # -- instances and reference objective values ---------------------------
    def instances(self) -> List[Dict[str, Any]]:
        if self._instances is None:
            grid = list(INSTANCE_GRID)[: self.n_instances]
            self._instances = [
                make_instance(self.seed + i, n, p, s, c, self.n_lambda)
                for i, (n, p, s, c) in enumerate(grid)
            ]
        return self._instances

    def reference(self) -> Dict[str, List[float]]:
        """``F_k(w_k_sklearn)`` for every instance -- the correctness yardstick."""
        if self._reference is None:
            from sklearn.linear_model import lasso_path

            ref: Dict[str, List[float]] = {}
            for inst in self.instances():
                _, coefs, _ = lasso_path(inst["X"], inst["y"], alphas=inst["lambdas"])
                W = np.asarray(coefs).T
                ref[inst["name"]] = [objective(inst["X"], inst["y"], W[k], float(inst["lambdas"][k]))
                                     for k in range(inst["lambdas"].size)]
            self._reference = ref
        return self._reference

    @property
    def timing_names(self) -> List[str]:
        return [i["name"] for i in self.instances()[-self.n_timing:]]

    @property
    def correctness_names(self) -> List[str]:
        return [i["name"] for i in self.instances()[: -self.n_timing]]

    def problem_statement(self) -> str:
        return (
            "# Lasso regularization path\n\n"
            "Given X in R^{n x p}, y in R^n and a decreasing lambda_1 > ... > lambda_K, define\n\n"
            "    F_k(w) = (1 / 2n) ||y - X w||_2^2 + lambda_k ||w||_1\n\n"
            "Return W = (w_1, ..., w_K).  The candidate passes if\n\n"
            f"    F_k(w_k) <= F_k(w_k_sklearn) + {self.tol:g}   for every k,\n\n"
            "checked on instances distinct from the timing ones.  The search score is the\n"
            "inverse geometric mean of the path times over the timing instances.\n\n"
            f"Write `{self.eval_program}` exposing EITHER\n"
            "  * `CPP_CODE` (a complete C++ program) and optionally `COMPILE_FLAGS`, driven\n"
            "    over stdin/stdout with the binary protocol in the task docstring, OR\n"
            "  * `solve_path(X, y, lambdas) -> W` of shape (K, p).\n"
        )

    def baseline_source(self) -> str:
        return BASELINE

    # -- evaluation ----------------------------------------------------------
    def evaluate(self, node_dir: str) -> EvalResult:
        path = os.path.join(node_dir, self.eval_program)
        if not os.path.exists(path):
            return EvalResult(fail_class="compile_other", error=f"missing {self.eval_program}")

        module, err = _import_candidate(path)
        if module is None:
            return EvalResult(fail_class="compile_other", error=err)

        cpp = getattr(module, "CPP_CODE", None)
        if cpp:
            workdir = os.path.join(node_dir, "build")
            os.makedirs(workdir, exist_ok=True)
            flags = list(getattr(module, "COMPILE_FLAGS", []) or [])
            binary, log = compile_native(cpp, flags, workdir, compiler=self.compiler,
                                         eigen_include=self.eigen_include)
            if binary is None:
                return EvalResult(fail_class="compile_other",
                                  error=log[-4000:], diagnostics={"route": "native"})
            overhead = launch_overhead(binary, self.timeout) if self.subtract_overhead else 0.0
            runner = lambda inst: run_native(binary, inst, self.timeout, overhead)  # noqa: E731
            route = "native"
        elif hasattr(module, "solve_path"):
            runner = lambda inst: run_python_solver(module, inst)  # noqa: E731
            route = "python"
        else:
            return EvalResult(fail_class="compile_other",
                              error="method.py must define CPP_CODE or solve_path(X, y, lambdas)")

        ref = self.reference()
        checked = 0
        for inst in self.instances():
            if inst["name"] not in self.correctness_names:
                continue
            try:
                W, _ = runner(inst)
            except subprocess.TimeoutExpired:
                return EvalResult(fail_class="timeout", valid=False, score=0.0,
                                  n_valid=checked, n_total=len(self.correctness_names),
                                  error=f"{inst['name']}: exceeded {self.timeout:g}s")
            except Exception as exc:  # noqa: BLE001
                return EvalResult(fail_class="runtime", valid=False, score=0.0,
                                  n_valid=checked, n_total=len(self.correctness_names),
                                  error=f"{inst['name']}: {type(exc).__name__}: {exc}")
            bad = _objective_check(inst, W, ref[inst["name"]], self.tol)
            if bad is not None:
                return EvalResult(fail_class="correctness", valid=False, score=0.0,
                                  n_valid=checked, n_total=len(self.correctness_names),
                                  error=f"{inst['name']}: {bad}",
                                  diagnostics={"route": route})
            checked += 1

        times: List[float] = []
        per_instance: Dict[str, float] = {}
        for inst in self.instances():
            if inst["name"] not in self.timing_names:
                continue
            samples = []
            for _ in range(max(1, self.repeats)):
                try:
                    _, elapsed = runner(inst)
                except Exception as exc:  # noqa: BLE001
                    return EvalResult(fail_class="runtime", valid=False, score=0.0,
                                      error=f"timing {inst['name']}: {type(exc).__name__}: {exc}")
                samples.append(elapsed)
            t = statistics.median(samples)
            per_instance[inst["name"]] = t
            times.append(max(t, 1e-9))

        score = float(np.exp(-np.mean(np.log(times)))) if times else 0.0
        diagnostics = {"route": route, "R_search": score, "times_s": per_instance,
                       "mean_path_ms": 1000.0 * float(np.mean(times))}
        if route == "native":
            diagnostics["launch_overhead_s"] = overhead
        return EvalResult(score=score, valid=True, fail_class="ok",
                          n_valid=checked, n_total=len(self.correctness_names),
                          diagnostics=diagnostics)


def _objective_check(inst, W: np.ndarray, ref: List[float], tol: float) -> Optional[str]:
    lam = inst["lambdas"]
    if W.shape != (lam.size, inst["X"].shape[1]):
        return f"expected W of shape {(lam.size, inst['X'].shape[1])}, got {W.shape}"
    if not np.all(np.isfinite(W)):
        return "coefficients contain non-finite values"
    for k in range(lam.size):
        got = objective(inst["X"], inst["y"], W[k], float(lam[k]))
        if got > ref[k] + tol:
            return (f"objective check failed at k={k}: "
                    f"F_k(w) = {got:.9f} > {ref[k]:.9f} + {tol:g}")
    return None


def _import_candidate(path: str):
    import importlib.util

    name = "dreamrsi_lasso_candidate_" + str(abs(hash(path)) % (10 ** 8))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        return None, f"cannot import {path}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001
        sys.modules.pop(name, None)
        return None, f"{type(exc).__name__}: {exc}"
    return module, ""


register("lasso", LassoPathTask)
