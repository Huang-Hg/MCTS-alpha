"""
Subtree-hash → ndarray 内存缓存,**dense-pack 稀疏存储**。

存储策略:
    - 条目打包:(bitmap u64 行对齐, values f64 紧凑, row_off i64, S)。alpha 整列
      NaN 占比高(symbol 上市跨度 + ts 暖机头),打包后条目 ~0.2-0.4× 原大小
      → 同 byte cap 容纳 3-6× 条目,命中率随容量涨(LRU 容量瓶颈是命中率根因)。
    - **无损**:非 NaN 值 memcpy 级逐位还原(含 ±inf/-0.0);NaN 还原为 canonical
      NAN(下游只 isnan,payload 无读者)。hit 返回 unpack 的新数组(非同一对象)。
    - **LRU 驱逐**:每次 hit / set 把 key 挪到 OrderedDict 末尾,超限弹队头。
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Tuple

import numpy as np

from backtest import ops


class EvalCache:
    """键 = node.hash (int64),值 = (T,S) float64 ndarray(内部打包存)。bytes / entries 双上限。"""

    def __init__(self, max_entries: int = 200_000, max_bytes: int = 2 * 1024**3):
        self._d: "OrderedDict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, int]]" = OrderedDict()
        self._max_entries = max_entries
        self._max_bytes = max_bytes
        self._bytes = 0
        self._seen: dict = {}          # hash → 请求次数(块化求值器准入:二次使用才物化)
        self.hits = 0
        self.misses = 0

    def bump(self, key: int) -> int:
        """记一次对 key 的请求,返回累计次数。块化求值器对 miss 的链内点调用:
        首次请求不物化(流式块过),二次起物化整列 + 入 cache(详见 expression._prepare)。
        _seen 只存 int→int,超 1M 条(~80MB)整体重置,准入计数从零再热。"""
        if len(self._seen) > 1_000_000:
            self._seen.clear()
        c = self._seen.get(key, 0) + 1
        self._seen[key] = c
        return c

    @staticmethod
    def _entry_bytes(e: tuple) -> int:
        bm, vals, off, _ = e
        return int(bm.nbytes + vals.nbytes + off.nbytes)

    def __contains__(self, key: int) -> bool:
        if key in self._d:
            self._d.move_to_end(key)
            self.hits += 1
            return True
        self.misses += 1
        return False

    def __getitem__(self, key: int) -> np.ndarray:
        bm, vals, off, S = self._d[key]
        self._d.move_to_end(key)
        return ops.unpack_sparse(bm, vals, off, S)

    def __setitem__(self, key: int, val: np.ndarray) -> None:
        bm, vals, off = ops.pack_sparse(val)
        entry = (bm, vals, off, int(val.shape[1]))
        sz = self._entry_bytes(entry)
        while self._d and (len(self._d) >= self._max_entries
                           or self._bytes + sz > self._max_bytes):
            _, e0 = self._d.popitem(last=False)          # 队头 = 最久未用
            self._bytes -= self._entry_bytes(e0)
        self._d[key] = entry
        self._d.move_to_end(key)
        self._bytes += sz

    def stats(self) -> dict:
        return {'size': len(self._d), 'bytes': self._bytes,
                'hits': self.hits, 'misses': self.misses}

    def clear(self) -> None:
        self._d.clear()
        self._bytes = 0
        self._seen.clear()
        self.hits = 0
        self.misses = 0
