# -*- coding: utf-8 -*-
"""手写数字分类的超参数优化问题：4 维混合变量，5 折交叉验证错误率。"""
import warnings

import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# x1: log10(学习率), x2: log10(L2 系数), x3: 第一隐藏层, x4: 第二隐藏层
BOUNDS = np.array([[-5.0, -1.0], [-6.0, -1.0], [16.0, 256.0], [8.0, 128.0]])
N_SPLITS = 5
MAX_ITER = 100

X, Y = load_digits(return_X_y=True)
X = X.astype(np.float64) / 16.0  # 像素值缩到 0~1，MLP 收敛更稳

_splits_cache = {}
_eval_cache = {}
cache_hits = 0


def get_splits(seed):
    """同一个 run 内所有算法共用同一组分层折分。"""
    if seed not in _splits_cache:
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        _splits_cache[seed] = list(skf.split(X, Y))
    return _splits_cache[seed]


def decode(x):
    """把实数向量解码成 MLP 的超参数。"""
    lr = float(10.0 ** np.clip(x[0], BOUNDS[0, 0], BOUNDS[0, 1]))
    alpha = float(10.0 ** np.clip(x[1], BOUNDS[1, 0], BOUNDS[1, 1]))
    h1 = int(np.clip(np.round(x[2]), BOUNDS[2, 0], BOUNDS[2, 1]))
    h2 = int(np.clip(np.round(x[3]), BOUNDS[3, 0], BOUNDS[3, 1]))
    return lr, alpha, h1, h2


def clear_eval_cache():
    """每个算法任务开始前清空评价缓存，避免同分片内的算法共用前序缓存。"""
    global _eval_cache, cache_hits
    _eval_cache = {}
    cache_hits = 0


def get_cache_hits():
    return cache_hits


def evaluate(x, seed):
    """返回 5 折平均验证错误率；同一个 run 内重复参数直接查缓存。"""
    global cache_hits
    lr, alpha, h1, h2 = decode(x)
    key = (seed, round(lr, 12), round(alpha, 12), h1, h2)
    if key in _eval_cache:
        cache_hits += 1
        return _eval_cache[key]

    errors = []
    for train_idx, test_idx in get_splits(seed):
        clf = MLPClassifier(
            hidden_layer_sizes=(h1, h2),
            activation="relu",
            solver="adam",
            alpha=alpha,
            learning_rate_init=lr,
            max_iter=MAX_ITER,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=5,
            tol=1e-4,
            random_state=0,
        )
        clf.fit(X[train_idx], Y[train_idx])
        pred = clf.predict(X[test_idx])
        errors.append(1.0 - accuracy_score(Y[test_idx], pred))

    value = float(np.mean(errors))
    _eval_cache[key] = value
    return value


def evaluate_many(pop, seed):
    return np.array([evaluate(ind, seed) for ind in pop], dtype=float)


def initial_population(seed, size):
    """所有算法共用的初始采样点。"""
    rng = np.random.default_rng([seed, 0])
    return rng.uniform(BOUNDS[:, 0], BOUNDS[:, 1], size=(size, len(BOUNDS)))
