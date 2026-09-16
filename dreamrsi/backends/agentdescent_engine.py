"""Running the online stage on ``agentdescent.dream.explore``.

The two packages disagree about **who drives the loop**, and that is the whole
difficulty of putting one engine under the other.

``dreamrsi``'s policy drives.  Appendix B.2 -- which this repository reproduces
verbatim, whose constraints :mod:`dreamrsi.policy.loader` enforces statically,
and which every shipped policy is written against -- gives the policy a
``solve(question, budget)`` that loops on its own and calls
``question.probe_batch(...)`` when it wants a batch revealed.

``agentdescent``'s engine drives.  ``explore()`` owns the round loop and asks
``policy.select(ctx, n)`` for one batch each time round.

Both are reasonable and neither may change: the policy API is the paper's, and
the engine is the thing being adopted.  So this module inverts control instead.
``solve()`` runs on a worker thread and is allowed to believe it is still
driving; each ``probe_batch`` hands its batch to ``select()`` through a
rendezvous and blocks until the engine has run it.  From the policy's side
nothing has changed, which is the point -- the same ``solve()`` that runs under
:mod:`dreamrsi.online` runs here, and the existing tests are the proof.

Three details make the substitution exact rather than merely similar:

* :class:`_BridgeWorld` subclasses :class:`dreamrsi.world.BaseWorld`, so batch
  validation, cost accounting and the round log are *the same code*, not a
  reimplementation that happens to agree today;
* the per-attempt work -- resume the parent workspace, call the agent, evaluate,
  write ``eval/score.json`` -- is :meth:`dreamrsi.online.LiveWorld._run_one`,
  reused rather than restated;
* nodes are appended to the tree serially in batch order, and every node in a
  batch takes the sequence number the tree had when the batch began, which is
  what the native engine does by adding after its parallel map returns.
"""

from __future__ import annotations

import json
import os
import queue
import threading
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..agents.base import DiscoveryAgent
from ..online import LiveWorld, live_cycle_manifest
from ..policy.api import finalize_result
from ..tasks.base import Task
from ..tree import DiscoveryTree
from ..types import ROOT_ID, CellMeta, GridPlan, Observation, SimResult, cell_id
from . import require_agentdescent

#: Passed through the rendezvous to unwind whichever side is still waiting.
_STOP = object()


class _Cancelled(RuntimeError):
    """Raised inside ``solve()`` when the engine ended the rollout under it."""


class _BridgeWorld(LiveWorld):
    """The ``question`` a policy sees while ``agentdescent`` drives the loop.

    It is a :class:`~dreamrsi.online.LiveWorld` with one method replaced. The
    legal sets, the grid, ``meta()``, the per-attempt work and all of
    :class:`~dreamrsi.world.BaseWorld`'s accounting are inherited unchanged; only
    the transition differs, because here the batch is run by the engine rather
    than by this object's own thread pool.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Unbounded on purpose. The turn-taking is enforced by the get/put
        # pairing below, and a bounded queue would add a second way to block:
        # the policy thread's final "I am done" would wait on a batch nobody is
        # coming back for.
        #: ``probe_batch`` -> ``select``: the batch the policy wants revealed.
        self._wanted: "queue.Queue[Any]" = queue.Queue()
        #: ``select`` -> ``probe_batch``: that batch has been run.
        self._revealed: "queue.Queue[Any]" = queue.Queue()
        #: Nodes the engine produced this round, by cell, awaiting the tree.
        self._produced: Dict[str, Any] = {}
        self._produced_lock = threading.Lock()

    def probe_batch(self, cells: Sequence[str],
                    on_reveal: Optional[Callable[[Observation], Any]] = None
                    ) -> List[Observation]:
        batch = self._validate(cells)
        if not batch:
            return []
        prefix_size = len(self._observed)
        self._wanted.put(list(batch))
        answer = self._revealed.get()
        if answer is _STOP:
            raise _Cancelled("the engine ended this rollout")
        if isinstance(answer, BaseException):
            raise answer

        # Serial, in batch order, sharing one sequence number -- which is what
        # the native engine produces by adding after its parallel map returns.
        seq = len(self.tree)
        with self._produced_lock:
            nodes = [self._produced.pop(cid) for cid in batch]
        for node in nodes:
            node.seq = seq
            self.tree.add(node)
            if self.on_node is not None:
                self.on_node(node)

        self._charge(batch)
        out: List[Observation] = []
        for cid in batch:
            obs = self.tree.observation_of(cid)
            self._record(obs)
            out.append(obs)
            if on_reveal is not None:
                on_reveal(obs)
        self._log_round(batch, out, prefix_size)
        return out


class _Driver:
    """An ``agentdescent`` ``SelectionPolicy`` backed by a ``dreamrsi`` ``solve()``.

    ``select`` is called by the engine once per round. It releases the previous
    batch's results to the policy thread, then blocks until that thread asks for
    the next batch, so the two loops take turns on one rendezvous instead of
    running independently.
    """

    def __init__(self, policy: Any, world: _BridgeWorld, budget: Optional[int]) -> None:
        self.policy, self.world, self.budget = policy, world, budget
        self.result: Any = None
        self.error: Optional[BaseException] = None
        self.requested: List[List[str]] = []
        self._thread: Optional[threading.Thread] = None
        self._outstanding = False
        #: Set once the engine has seen the policy's own "no more batches", so
        #: :meth:`finish` knows not to wait for a message already delivered.
        self._policy_done = False

    # -- the policy thread ---------------------------------------------------
    def _run(self) -> None:
        try:
            self.result = self.policy.solve(self.world, self.budget)
        except _Cancelled:
            pass
        except BaseException as exc:  # noqa: BLE001 - re-raised on the caller's thread
            self.error = exc
        finally:
            self.world._wanted.put(_STOP)

    def finish(self, timeout: float = 60.0) -> None:
        """Unwind the policy thread, whichever side stopped first.

        The engine's loop ends *after* running a batch and before asking for the
        next one, so the last batch's results are still owed to the policy. They
        are released first -- otherwise that round would be missing from the
        episode's own accounting -- and only a request that comes after it is
        refused.
        """
        if self._thread is None:
            return
        if self._outstanding:
            self.world._revealed.put(None)
            self._outstanding = False
        while not self._policy_done:
            item = self.world._wanted.get()
            if item is _STOP:
                self._policy_done = True
                break
            self.world._revealed.put(_STOP)
        self._thread.join(timeout)
        if self.error is not None:
            raise self.error

    # -- the engine's side ---------------------------------------------------
    def select(self, ctx, n: int):
        dream = require_agentdescent()
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="dreamrsi-policy",
                                            daemon=True)
            self._thread.start()
        elif self._outstanding:
            self.world._revealed.put(None)
            self._outstanding = False

        wanted = self.world._wanted.get()
        if wanted is _STOP:
            self._policy_done = True
            return []
        self.requested.append(list(wanted))
        self._outstanding = True

        # Every node carries its cell in ``detail``, which reaches the policy as
        # ``Candidate.state`` -- so the engine's own view is what resolves a
        # parent, with no index bookkeeping of ours to fall out of step.
        by_cell = {c.state.get("cell"): c for c in ctx.candidates
                   if c.state.get("cell")}
        picks = []
        for cid in wanted:
            branch, _, attempt = cid[1:].partition("a")
            parent_cell = (ROOT_ID if int(attempt) == 0
                           else cell_id(int(branch), int(attempt) - 1))
            found = by_cell.get(parent_cell)
            if found is None:  # pragma: no cover - a legal cell always has a parent
                raise KeyError(f"no engine node for {parent_cell!r}, parent of {cid!r}")
            picks.append(found)
        return picks


class _AttemptRunner:
    """``agentdescent``'s ``Continuation``, backed by the discovery agent.

    ``explore`` hands over a parent and a running attempt counter. The cell that
    attempt produces is the one the policy asked for, in the order it asked, so
    the counter's offset within the current batch names it -- and the batches
    already run give the offset's base.
    """

    def __init__(self, world: _BridgeWorld, driver: _Driver) -> None:
        self.world, self.driver = world, driver

    def __call__(self, tree, parent, attempt: int):
        dream = require_agentdescent()
        batch = self.driver.requested[-1]
        base = sum(len(b) for b in self.driver.requested[:-1])
        offset = attempt - base
        cid = batch[offset] if 0 <= offset < len(batch) else batch[-1]

        node, _ = self.world._run_one(cid)
        with self.world._produced_lock:
            self.world._produced[cid] = node
        return dream.Attempt(
            score=None if node.score == float("-inf") else float(node.score),
            valid=bool(node.valid),
            detail={"cell": cid, "fail_class": node.fail_class,
                    "error": node.error or ""},
        )


def run_online_cycle(task: Task, agent: DiscoveryAgent, policy, plan: GridPlan, *,
                     run_dir: str, history_dir: str, baseline_dir: str,
                     problem_file: str, baseline_score: float,
                     max_parallelism: int = 4, max_decision_rounds: int = 64,
                     budget: Optional[int] = None, world_id: str = "live",
                     direction_guidance: str = "", max_workers: Optional[int] = None,
                     cycle: int = 1) -> Dict[str, Any]:
    """:func:`dreamrsi.online.run_online_cycle`, with ``explore()`` as the engine.

    Same arguments, same return value, same files on disk. What differs is that
    the round loop, the batch parallelism and the recorded tree are
    ``agentdescent``'s, and the policy reaches them through :class:`_Driver`.
    The extra ``engine_tree`` key is the world as ``agentdescent`` recorded it.
    """
    dream = require_agentdescent()
    os.makedirs(run_dir, exist_ok=True)

    world = _BridgeWorld(task, agent, plan, run_dir=run_dir, history_dir=history_dir,
                         baseline_dir=baseline_dir, problem_file=problem_file,
                         baseline_score=baseline_score,
                         max_parallelism=max_parallelism,
                         max_decision_rounds=max_decision_rounds, world_id=world_id,
                         direction_guidance=direction_guidance,
                         max_workers=max_workers, cycle=cycle)
    driver = _Driver(policy, world, budget)

    adt = dream.explore(
        _AttemptRunner(world, driver), driver,
        n_workers=max_parallelism,
        max_rounds=max_decision_rounds,
        root_score=baseline_score,
        root_detail={"cell": ROOT_ID, "task": task.name, "role": "baseline"},
        max_concurrency=int(max_workers or max_parallelism),
        name=world_id,
    )
    driver.finish()

    # A policy the engine outran never returned a result; close the episode with
    # the environment's own bookkeeping, as ``finalize_result`` does online.
    result = driver.result
    if result is None:
        result = finalize_result(world, SimResult(), "decision_round_cap")

    tree = world.tree
    tree.meta["decision_rounds"] = adt.decision_rounds
    tree.meta["engine"] = "agentdescent"
    trace_path = os.path.join(run_dir, "trace.json")
    tree.save(trace_path)

    manifest = live_cycle_manifest(tree, result, plan, world, policy=policy)
    with open(os.path.join(run_dir, "live_cycle_manifest.json"), "w",
              encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")
    return {"tree": tree, "result": result, "manifest": manifest,
            "trace_path": trace_path, "world": world, "engine_tree": adt}


__all__ = ["run_online_cycle"]
