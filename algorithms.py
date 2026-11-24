# -*- coding: utf-8 -*-
"""五种调参策略：随机搜索、实数 GA、LDIW-PSO、DE/rand/1/bin、jDE。"""
import numpy as np

import problems

BOUNDS = problems.BOUNDS
POP_SIZE = 20
MAX_FES = 400
N_GEN = MAX_FES // POP_SIZE - 1  # 初始 20 个点 + 19 代，每代 20 个点


def _tournament(pop, fit, rng, k=3):
    """k 元锦标赛，返回规模为 POP_SIZE 的父代集合。"""
    idx = rng.integers(0, POP_SIZE, size=(POP_SIZE, k))
    winners = np.argmin(fit[idx], axis=1)
    return pop[idx[np.arange(POP_SIZE), winners]]


def random_search(seed, initial):
    rng = np.random.default_rng([seed, 1])
    pop = initial.copy()
    fit = problems.evaluate_many(pop, seed)
    best_idx = int(np.argmin(fit))
    best_x, best = pop[best_idx].copy(), float(fit[best_idx])
    history = [best]

    for _ in range(N_GEN):
        batch = rng.uniform(BOUNDS[:, 0], BOUNDS[:, 1], size=(POP_SIZE, len(BOUNDS)))
        batch_fit = problems.evaluate_many(batch, seed)
        idx = int(np.argmin(batch_fit))
        if batch_fit[idx] < best:
            best = float(batch_fit[idx])
            best_x = batch[idx].copy()
        history.append(best)
    return {"best_error": best, "best_x": best_x, "history": np.array(history), "extra": None}


def real_ga(seed, initial):
    rng = np.random.default_rng([seed, 2])
    pop = initial.copy()
    fit = problems.evaluate_many(pop, seed)
    best_idx = int(np.argmin(fit))
    best_x, best = pop[best_idx].copy(), float(fit[best_idx])
    history = [best]

    span = BOUNDS[:, 1] - BOUNDS[:, 0]
    for gen in range(1, N_GEN + 1):
        parents = _tournament(pop, fit, rng, k=3)
        children = parents.copy()
        order = rng.permutation(POP_SIZE)
        for k in range(0, POP_SIZE, 2):
            i, j = order[k], order[k + 1]
            if rng.random() < 0.8:
                alpha = rng.random()
                children[i] = alpha * parents[i] + (1.0 - alpha) * parents[j]
                children[j] = alpha * parents[j] + (1.0 - alpha) * parents[i]

        # 变异步长从 0.15 区间逐渐收到 0.01 区间，后期才可能微调学习率
        sigma_ratio = 0.15 - 0.14 * (gen - 1) / max(N_GEN - 1, 1)
        mask = rng.random((POP_SIZE, len(BOUNDS))) < 0.1
        noise = rng.normal(0.0, sigma_ratio * span, size=(POP_SIZE, len(BOUNDS)))
        children = np.clip(children + mask * noise, BOUNDS[:, 0], BOUNDS[:, 1])

        child_fit = problems.evaluate_many(children, seed)
        worst = int(np.argmax(child_fit))
        if best < child_fit[worst]:
            children[worst] = best_x
            child_fit[worst] = best
        pop, fit = children, child_fit

        idx = int(np.argmin(fit))
        if fit[idx] < best:
            best = float(fit[idx])
            best_x = pop[idx].copy()
        history.append(best)
    return {"best_error": best, "best_x": best_x, "history": np.array(history), "extra": None}


def ldiw_pso(seed, initial):
    rng = np.random.default_rng([seed, 3])
    pos = initial.copy()
    vel = rng.uniform(-0.15, 0.15, size=pos.shape) * (BOUNDS[:, 1] - BOUNDS[:, 0])
    fit = problems.evaluate_many(pos, seed)

    pbest = pos.copy()
    pbest_fit = fit.copy()
    g_idx = int(np.argmin(pbest_fit))
    gbest, gbest_fit = pbest[g_idx].copy(), float(pbest_fit[g_idx])
    history = [gbest_fit]

    vmax = 0.15 * (BOUNDS[:, 1] - BOUNDS[:, 0])
    for gen in range(1, N_GEN + 1):
        w = 0.9 - 0.5 * (gen - 1) / max(N_GEN - 1, 1)
        r1 = rng.random(pos.shape)
        r2 = rng.random(pos.shape)
        vel = w * vel + 1.49445 * r1 * (pbest - pos) + 1.49445 * r2 * (gbest - pos)
        vel = np.clip(vel, -vmax, vmax)
        pos = np.clip(pos + vel, BOUNDS[:, 0], BOUNDS[:, 1])
        fit = problems.evaluate_many(pos, seed)

        better = fit < pbest_fit
        pbest[better] = pos[better]
        pbest_fit[better] = fit[better]
        idx = int(np.argmin(pbest_fit))
        if pbest_fit[idx] < gbest_fit:
            gbest_fit = float(pbest_fit[idx])
            gbest = pbest[idx].copy()
        history.append(gbest_fit)
    return {"best_error": gbest_fit, "best_x": gbest, "history": np.array(history), "extra": None}


def _rand_indices(rng):
    n = POP_SIZE
    r1 = rng.integers(0, n, n)
    r2 = rng.integers(0, n, n)
    r3 = rng.integers(0, n, n)
    for i in range(n):
        while r1[i] == i:
            r1[i] = rng.integers(0, n)
        while r2[i] == i or r2[i] == r1[i]:
            r2[i] = rng.integers(0, n)
        while r3[i] == i or r3[i] == r1[i] or r3[i] == r2[i]:
            r3[i] = rng.integers(0, n)
    return r1, r2, r3


def differential_evolution(seed, initial, adaptive=False):
    rng = np.random.default_rng([seed, 4])
    pop = initial.copy()
    fit = problems.evaluate_many(pop, seed)
    best_idx = int(np.argmin(fit))
    best_x, best = pop[best_idx].copy(), float(fit[best_idx])
    history = [best]

    if adaptive:
        F = np.full(POP_SIZE, 0.5)
        CR = np.full(POP_SIZE, 0.9)
        params_history = []
    else:
        F = np.full(POP_SIZE, 0.5)
        CR = np.full(POP_SIZE, 0.9)

    for _ in range(N_GEN):
        r1, r2, r3 = _rand_indices(rng)
        if adaptive:
            trial_F = F.copy()
            trial_CR = CR.copy()
            mask_f = rng.random(POP_SIZE) < 0.1
            mask_cr = rng.random(POP_SIZE) < 0.1
            trial_F[mask_f] = 0.1 + 0.9 * rng.random(mask_f.sum())
            trial_CR[mask_cr] = rng.random(mask_cr.sum())
        else:
            trial_F, trial_CR = F, CR

        mutant = pop[r1] + trial_F[:, None] * (pop[r2] - pop[r3])
        cross = rng.random(pop.shape) < trial_CR[:, None]
        cross[np.arange(POP_SIZE), rng.integers(0, pop.shape[1], POP_SIZE)] = True
        trial = np.clip(np.where(cross, mutant, pop), BOUNDS[:, 0], BOUNDS[:, 1])
        trial_fit = problems.evaluate_many(trial, seed)

        accept = trial_fit <= fit
        pop[accept] = trial[accept]
        fit[accept] = trial_fit[accept]
        if adaptive:
            F[accept] = trial_F[accept]
            CR[accept] = trial_CR[accept]
            params_history.append([float(F.mean()), float(CR.mean())])

        idx = int(np.argmin(fit))
        if fit[idx] < best:
            best = float(fit[idx])
            best_x = pop[idx].copy()
        history.append(best)

    extra = np.array(params_history) if adaptive else None
    return {"best_error": best, "best_x": best_x, "history": np.array(history), "extra": extra}
