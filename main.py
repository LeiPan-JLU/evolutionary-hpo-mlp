# -*- coding: utf-8 -*-
"""主实验：20 次独立运行，比较 5 种超参数搜索策略。"""
import csv
import json
import os
import subprocess
import sys
import time

# 限制每个进程的 BLAS 线程，避免多个进程互相抢核
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

import algorithms
import problems

N_RUNS = 20
POP_SIZE = 20
MAX_FES = 400
BASE_SEED = 2026
N_SHARDS = 20
ALGOS = ["RandomSearch", "RealGA", "LDIW-PSO", "DE", "jDE"]

RUNNERS = {
    "RandomSearch": algorithms.random_search,
    "RealGA": algorithms.real_ga,
    "LDIW-PSO": algorithms.ldiw_pso,
    "DE": lambda seed, initial: algorithms.differential_evolution(seed, initial, adaptive=False),
    "jDE": lambda seed, initial: algorithms.differential_evolution(seed, initial, adaptive=True),
}


def run_one(task):
    algo, run_id = task
    seed = BASE_SEED + run_id
    initial = problems.initial_population(seed, POP_SIZE)
    start = time.perf_counter()
    result = RUNNERS[algo](seed, initial)
    elapsed = time.perf_counter() - start

    lr, alpha, h1, h2 = problems.decode(result["best_x"])
    return {
        "algorithm": algo,
        "run": run_id + 1,
        "seed": seed,
        "best_error": result["best_error"],
        "best_lr": lr,
        "best_alpha": alpha,
        "best_h1": h1,
        "best_h2": h2,
        "time_sec": elapsed,
        "history": result["history"].tolist(),
        "jde_params": None if result["extra"] is None else result["extra"].tolist(),
    }


def run_shard(shard_id, n_shards):
    tasks = [(algo, run) for algo in ALGOS for run in range(N_RUNS)][shard_id::n_shards]
    items = []
    for i, task in enumerate(tasks, 1):
        res = run_one(task)
        items.append(res)
        print(f"[shard {shard_id:02d} {i}/{len(tasks)}] {res['algorithm']:12s} "
              f"run={res['run']:02d} err={res['best_error']:.4f} time={res['time_sec']:.1f}s",
              flush=True)
    with open(os.path.join("results", f"part_{shard_id:02d}.json"), "w", encoding="utf-8") as handle:
        json.dump({"items": items}, handle, ensure_ascii=False)


def run_parent():
    os.makedirs("results", exist_ok=True)
    start = time.perf_counter()
    procs = []
    for shard in range(N_SHARDS):
        env = os.environ.copy()
        env["EC_SHARD"] = str(shard)
        env["EC_SHARDS"] = str(N_SHARDS)
        procs.append(subprocess.Popen([sys.executable, os.path.abspath(__file__)], env=env))
    for proc in procs:
        if proc.wait() != 0:
            raise RuntimeError("a shard failed")

    items = []
    for shard in range(N_SHARDS):
        path = os.path.join("results", f"part_{shard:02d}.json")
        with open(path, encoding="utf-8") as handle:
            items.extend(json.load(handle)["items"])
        os.remove(path)

    csv_path = os.path.join("results", "raw_results.csv")
    fields = ["algorithm", "run", "seed", "best_error", "best_lr", "best_alpha",
              "best_h1", "best_h2", "time_sec"]
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for res in sorted(items, key=lambda r: (ALGOS.index(r["algorithm"]), r["run"])):
            writer.writerow({key: res[key] for key in fields})

    history = np.zeros((len(ALGOS), N_RUNS, MAX_FES // POP_SIZE))
    jde_params = np.zeros((N_RUNS, algorithms.N_GEN, 2))
    for res in items:
        ai = ALGOS.index(res["algorithm"])
        history[ai, res["run"] - 1] = res["history"]
        if res["jde_params"] is not None:
            jde_params[res["run"] - 1] = res["jde_params"]
    np.save(os.path.join("results", "convergence_history.npy"), history)
    np.save(os.path.join("results", "jde_params_history.npy"), jde_params)

    print(f"total wall time: {time.perf_counter() - start:.1f}s")
    print(f"saved: {csv_path}, results/convergence_history.npy, results/jde_params_history.npy")


def main():
    if "EC_SHARD" in os.environ:
        run_shard(int(os.environ["EC_SHARD"]), int(os.environ["EC_SHARDS"]))
    else:
        run_parent()


if __name__ == "__main__":
    main()
