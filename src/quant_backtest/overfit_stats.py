"""防过拟合统计纯函数库（防过拟合统计纯函数库）。

来源快照：
  - accounts.py (commit bd4df853ac56df1bb3dc389a8f9cd7005447efa7)
  - combo_opt_plateau_scan.py::eval_plateau_2d

仅依赖 Python 标准库，可直接用于离线或 NAS 单文件复用：
  config_hash / register_trials / mean_ex_top / bh_fdr / perm_pvalue /
  block_bootstrap_diff / eval_plateau_2d
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
from typing import Any, Mapping, Sequence


def config_hash(cfg: Any) -> str:
    """配置对象的稳定 SHA256 前 12 位哈希（按键排序、保留非 ASCII）。"""
    payload = json.dumps(cfg, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def register_trials(
    path: str,
    cfgs: Sequence[Any],
    tag: str,
    now: str | None = None,
) -> tuple[int, int]:
    """追加登记试验至 JSONL 台账；同哈希已登记则去重跳过。

    返回 ``(新登记数, 登记表中总试验数 N)``。
    """
    seen: set[str] = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for ln in f:
                if ln.strip():
                    seen.add(json.loads(ln)["hash"])
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    new = 0
    ts = now or time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with open(path, "a", encoding="utf-8") as f:
        for c in cfgs:
            h = config_hash(c)
            if h in seen:
                continue
            seen.add(h)
            new += 1
            f.write(
                json.dumps(
                    {"hash": h, "tag": tag, "ts": ts, "cfg": c},
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
    return new, len(seen)


def mean_ex_top(rs: Sequence[float], k: int = 5) -> float | None:
    """去掉最好 k 笔后的均值；样本数 <= k 时返回 None。"""
    if len(rs) <= k:
        return None
    trimmed = sorted(rs, reverse=True)[k:]
    return sum(trimmed) / len(trimmed)


def bh_fdr(pvals: Sequence[float], q: float = 0.05) -> list[bool]:
    """Benjamini-Hochberg 多重检验校正；返回与输入同序的显著性布尔列表。"""
    n = len(pvals)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: pvals[i])
    cut = -1
    for rank, i in enumerate(order, 1):
        if pvals[i] <= q * rank / n:
            cut = rank
    keep = [False] * n
    for rank, i in enumerate(order, 1):
        if rank <= cut:
            keep[i] = True
    return keep


def perm_pvalue(observed: float, null_samples: Sequence[float]) -> float:
    """单侧置换检验 p 值：``(count(null >= observed) + 1) / (N + 1)``。"""
    c = sum(1 for x in null_samples if x >= observed)
    return (c + 1) / (len(null_samples) + 1)


def block_bootstrap_diff(
    blocks_a: Mapping[Any, tuple[float, int]],
    blocks_b: Mapping[Any, tuple[float, int]],
    n: int = 2000,
    seed: int = 0,
    q: float = 0.05,
) -> tuple[float | None, float | None, float | None]:
    """按共享时间块重抽样估计两规则每笔均值差 (a - b) 的双侧置信区间。

    ``blocks_*`` 为 ``{块键: (R 之和, 笔数)}``。返回 ``(点估计, 下限, 上限)``。
    """
    keys = sorted(set(blocks_a) | set(blocks_b))
    rnd = random.Random(seed)
    zero = (0.0, 0)

    def est(ks: Sequence[Any]) -> float | None:
        sa = sb = 0.0
        na = nb = 0
        for k in ks:
            xa = blocks_a.get(k, zero)
            xb = blocks_b.get(k, zero)
            sa += xa[0]
            na += xa[1]
            sb += xb[0]
            nb += xb[1]
        return (sa / na - sb / nb) if na and nb else None

    point = est(keys)
    if not keys:
        return point, None, None
    out: list[float] = []
    for _ in range(n):
        v = est([keys[rnd.randrange(len(keys))] for _ in keys])
        if v is not None:
            out.append(v)
    if not out:
        return point, None, None
    out.sort()
    lo_idx = int(len(out) * q / 2)
    hi_idx = min(len(out) - 1, int(len(out) * (1 - q / 2)))
    return point, out[lo_idx], out[hi_idx]


def eval_plateau_2d(
    grid_stats: Mapping[tuple[int, int], Mapping[str, float] | None],
    nh: int,
    nc: int,
) -> dict[tuple[int, int], dict[str, Any]]:
    """二维正交邻域参数高原检验（需 >=2/3 有效正交邻域同号且 |mean| >= |center|/2）。"""
    out: dict[tuple[int, int], dict[str, Any]] = {}
    for hi_i in range(nh):
        for ci_i in range(nc):
            cst = grid_stats.get((hi_i, ci_i))
            if cst is None:
                out[(hi_i, ci_i)] = {
                    "ok": None,
                    "good_nb": 0,
                    "valid_nb": 0,
                    "pos_nb": 0,
                    "lo_pos_nb": 0,
                }
                continue
            cmean = cst["mean"]
            nbs = []
            for dh, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nhi, nci = hi_i + dh, ci_i + dc
                if 0 <= nhi < nh and 0 <= nci < nc and grid_stats.get((nhi, nci)) is not None:
                    nbs.append(grid_stats[(nhi, nci)])
            if not nbs:
                out[(hi_i, ci_i)] = {
                    "ok": None,
                    "good_nb": 0,
                    "valid_nb": 0,
                    "pos_nb": 0,
                    "lo_pos_nb": 0,
                }
                continue
            good = sum(
                1
                for nb in nbs
                if nb["mean"] * cmean > 0 and abs(nb["mean"]) >= abs(cmean) / 2.0
            )
            pos_nb = sum(1 for nb in nbs if nb["mean"] > 0)
            lo_pos_nb = sum(1 for nb in nbs if nb.get("lo", 0.0) > 0)
            ok = cmean > 0 and (good >= (2.0 / 3.0) * len(nbs))
            out[(hi_i, ci_i)] = {
                "ok": ok,
                "good_nb": good,
                "valid_nb": len(nbs),
                "pos_nb": pos_nb,
                "lo_pos_nb": lo_pos_nb,
            }
    return out
