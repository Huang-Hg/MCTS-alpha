"""项目入口 — 训练 / 物化(quant-server 跑)。

每个子命令只接位置参数:输出路径(materialize 要输入 JSON)。
所有时间区间 / 超参 / 数据路径 / 策略网开关 / 缓存开关都在 `config/config.ini`。
改 INI 即改全局,不再有 CLI 覆盖分支。

用法:
    python main.py gp-baseline   OUTPUT_JSON      # DEAP 强类型 GP 因子挖掘(NSGA-II)+ alpha pool
    python main.py alphasage     OUTPUT_JSON      # AlphaSAGE GFlowNet 因子挖掘 + alpha pool
    python main.py materialize   INPUT_JSON   OUTPUT_DIR

实盘(binance-trader 跑)走独立入口 `python -m trade`,见 trade/__main__.py。
sizing 已是确定性(AFF 融合 + topk + vol-target,见 trade/signal.py);旧 RL three-head policy 已删(2026-06-27)。

gp-baseline / alphasage = 搜索 + net 感知准入建池(AFF 因果滚动融合);评估全复用 evaluator/alpha_pool。
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import logging
import os
import signal
import sys
import time
from calendar import monthrange
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

# kill -SIGUSR1 PID → 把全 Python thread 栈打到 stderr(诊断 deadlock/hang;不停进程)
# faulthandler.register + SIGUSR1 仅 Unix 有(OS 系统边界差异);Windows 本地无 → 跳过(server Linux 照常注册)。
if hasattr(faulthandler, 'register') and hasattr(signal, 'SIGUSR1'):
    faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)

# libgomp 空闲线程默认 spin-wait(ACTIVE):self-play worker 的 OMP 线程在主进程**串行训练段**仍
#   空转抢核(实测训练段 worker 占 6-9 核)→ tree-LSTM 训练被饿死 ~40×(2026-06-22 实锤 train
#   290s→7.7s,sp 无损)。passive=空闲即睡。**必须在 torch/numpy/C 扩展(libgomp init)导入前设。**
os.environ.setdefault('OMP_WAIT_POLICY', 'passive')
os.environ.setdefault('GOMP_SPINCOUNT', '0')

import numpy as np

logging.Formatter.converter = time.gmtime
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)sZ %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
log = logging.info

from config.config import FactorMiningConfig, ini
from backtest import ops
from evaluation.adapter import load_panel, month_iter, split_train_val
from evaluation.ast import AlphaTree
from evaluation.expression import evaluate as eval_tree
from evaluation.grammar import OperandToken
from rl.alpha_pool import AlphaPool
from rl.backtest_reward import BacktestRewardConfig
from rl.evaluator import _DEVICE as _EVAL_DEVICE, EvalConfig, evaluate_alpha
from backtest.ops import cs_zscore_np, icir
from evaluation.cache import EvalCache


# ============================================================================
# gp-baseline — DEAP 强类型 GP 因子挖掘(NSGA-II 双目标;评估全复用共享管线)
#   数据/切分/因果 universe/1h 降采样/val 池回报口径 = 冻结口径,逐字一致。
# ============================================================================

# 数据切分冻结口径(全库唯一,2026-06-06 定死):train 24m + val 10m;test 完全 held out。
_GP_TRAIN = ('2023-03-01', '2025-02-28')
_GP_VAL   = ('2025-03-01', '2025-04-30')   # 2026-06-26 拥挤中性重训:val 仅 2 月(降本地 load 内存;test 2026-01~05 仍 held out)


def prepare_gp_data(train_start: str, train_end: str, val_start: str, val_end: str,
                    parquet_root: str, primary_horizon: int = 24) -> Dict[str, Any]:
    """加载 + 预处理 GP 搜索/评估数据(口径同原 train-policy):因果 universe y-NaN-mask、train/val
    切分、operand 面板降 1h(ts O(T) 缩 12×)、val 5m 价路径辅助(run_bt 用)+ 1h dec 诊断网格。"""
    import gc
    from rl.backtest_reward import prepare_close_TS, prepare_open_TS
    from evaluation.aggregator import aggregate_5m_to_1h
    bundle = load_panel(train_start=train_start, train_end=train_end,
                        val_start=val_start, val_end=val_end, parquet_root=parquet_root,
                        primary_horizon=primary_horizon)
    bundle.highs = bundle.lows = None
    # 因果截面:y 一次 NaN-mask 到 valid = listed(qvol>0)∧ member → IC/per_t_pnl 严限因果 universe
    valid_full = (bundle.bar_quote_volume > 0.0) & bundle.member_mask
    bundle.y_future[~valid_full] = np.nan
    _tr_idx = np.where(bundle.train_mask)[0]
    _va_idx = np.where(bundle.val_mask)[0]
    _va_sl = slice(int(_va_idx[0]), int(_va_idx[-1]) + 1)
    val_valid = valid_full[_va_sl].copy()
    del valid_full
    _T_train_rows = len(_tr_idx)
    train_panels, train_y, val_panels, val_y = split_train_val(bundle)
    # val 5m 价路径(run_bt;须在 val_panels 降 1h 前取)
    val_close_TS = prepare_close_TS(val_panels[OperandToken.CLOSE])
    val_open_TS  = prepare_open_TS(bundle.opens[_va_sl])
    val_fund_TS  = bundle.funding_rate[_va_sl].copy()
    val_vol_TS   = bundle.slippage_vol[_va_sl].copy()
    val_qvol_TS  = bundle.bar_quote_volume[_va_sl].copy()
    _T_log, _S_log = bundle.T, bundle.S
    val_syms = list(bundle.symbols)             # (S,) symbol 名,供 per-symbol 诊断(纯增量,训练不消费)
    del bundle                                  # 全 5m aux(opens/funding/vol/qvol 全长)用完即弃,val TS 已 copy
    # operand 面板降 1h;y 取 block-start 行(2h 前向收益)。聚合后 5m train/val 拷贝随重绑定释放 →
    # 训练只用 1h,5m 不在训练期常驻(gc 归还 OS,峰值仅在此处过渡区)。
    train_panels = aggregate_5m_to_1h(train_panels)
    train_y = np.ascontiguousarray(train_y[::12][:_T_train_rows // 12])
    val_panels = aggregate_5m_to_1h(val_panels)
    val_y = np.ascontiguousarray(val_y[::12][:len(_va_idx) // 12])
    gc.collect()
    val_dec   = np.arange(0, len(_va_idx) - 14, 12, dtype=np.int64) // 12
    val_y_dec = np.ascontiguousarray(val_y[val_dec])
    log(f"[gp-baseline] panel T={_T_log} S={_S_log} train_rows={_T_train_rows} val_rows={len(_va_idx)} "
        f"valid/bar val={val_valid.sum(1).mean():.1f}")
    return dict(train_panels=train_panels, train_y=train_y, val_panels=val_panels, val_y=val_y,
                val_dec=val_dec, val_y_dec=val_y_dec, val_valid=val_valid,
                val_close_TS=val_close_TS, val_open_TS=val_open_TS, val_fund_TS=val_fund_TS,
                val_vol_TS=val_vol_TS, val_qvol_TS=val_qvol_TS, val_syms=val_syms)


def eval_pool_val(pool, data: Dict[str, Any], bcfg) -> Tuple[float, float]:
    """池在 val 段的 ensemble 费后回报:成员 z-score →(可选 crowding 中性)→ AFF 因果滚动 lstsq 融合
    → 1h dec ensemble → β 中性 → top-K 多空构造(top_k/swap_n/min_hold,gross=1)→ broadcast 5m →
    run_bt(raw_weights)。返回 (pool_val_ic, pool_val_ret)。空池 → (0,0)。"""
    from rl.backtest_reward import run_bt, broadcast_dec_to_5m
    from rl.sizing import (topk_ls_weights, beta_neutralize, aff_fuse,
                           crowding_neutralize, build_crowding_basis, _CROWD_TOKENS)
    from rl.evaluator import EvalConfig
    if not pool.members:
        return 0.0, 0.0
    vd = data['val_dec']; n_dec = len(vd)
    valid_dec = data['val_valid'][::12][:n_dec]
    crowd = EvalConfig().crowding_neutral
    gz = (build_crowding_basis({t: np.ascontiguousarray(data['val_panels'][t], dtype=np.float64)[vd]
                                for t in _CROWD_TOKENS}) if crowd else None)
    member_z = []
    for m in pool.members:
        z = cs_zscore_np(np.ascontiguousarray(eval_tree(m.tree, data['val_panels'], None), dtype=np.float64)[vd])
        member_z.append(crowding_neutralize(z, gz) if crowd else z)             # 方向池:成员先拥挤中性再融合
    mz = np.stack(member_z, axis=2)                                            # (n_dec, S, K)
    # AFF 自适应融合(因果滚动 lstsq;取代静态 ICIR 加权 —— 2026-06-26 实证大胜弱方向因子);短窗自适应 warmup
    ens = aff_fuse(mz, data['val_y_dec'], valid_dec, warmup=min(480, n_dec // 4))
    _, pool_ic, _ = icir(ens, data['val_y_dec'])
    close_dec = np.ascontiguousarray(data['val_panels'][OperandToken.CLOSE], dtype=np.float64)[vd]
    ens = beta_neutralize(ens, close_dec, valid_dec)                           # 动态 β 中性(系统风险正交)
    w_dec = topk_ls_weights(ens, valid_dec, bcfg.top_k, bcfg.swap_n, bcfg.min_hold, leverage_cap=1.0)
    sig5m = broadcast_dec_to_5m(w_dec, data['val_close_TS'].shape[0])
    res = run_bt(sig5m, data['val_close_TS'], data['val_open_TS'], data['val_qvol_TS'],
                 data['val_vol_TS'], data['val_fund_TS'], bcfg,
                 valid_mask=data['val_valid'], raw_weights=True)
    return float(pool_ic), float(res['total_return'])


def cmd_gp_baseline(output: str) -> None:
    """DEAP 强类型 GP 因子挖掘:load panel → NSGA-II 搜索(train)→ net 感知准入建池 → val 池回报
    → 落 deploy bundle(output JSON)+ snapshot。device 由 [evaluator] 派发(V100 走 cuda 单进程)。"""
    os.environ['ARROW_DEFAULT_MEMORY_POOL'] = 'system'   # Arrow 还内存(同 train-policy)
    from search.gp.gp_baseline import run_gp
    fm = FactorMiningConfig()
    bcfg = BacktestRewardConfig()
    log(f"[gp-baseline] train={_GP_TRAIN[0]}~{_GP_TRAIN[1]}  val={_GP_VAL[0]}~{_GP_VAL[1]}  (test held out)")
    data = prepare_gp_data(_GP_TRAIN[0], _GP_TRAIN[1], _GP_VAL[0], _GP_VAL[1], fm.parquet_5m_root)
    pool, collected = run_gp(data['train_panels'], data['train_y'],
                             capacity=fm.pool_capacity, seed=42, log=log)
    pool_ic, pool_ret = eval_pool_val(pool, data, bcfg)
    log(f"[gp-baseline] pool_size={pool.size} collected={len(collected)} "
        f"val pool_ic={pool_ic:+.4f} pool_ret={pool_ret:+.4f}")
    run_ts = os.environ.get('RUN_TS') or time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    snap = Path(f'logs/gp_pool_{run_ts}.jsonl')
    snap.parent.mkdir(parents=True, exist_ok=True)
    with open(snap, 'w', encoding='utf-8') as f:
        f.write(json.dumps({'pool': pool.to_jsonable(), 'val_pool_ic': pool_ic,
                            'val_pool_ret': pool_ret, 'collected': len(collected)},
                           ensure_ascii=False) + '\n')
    with open(output, 'w', encoding='utf-8') as f:
        json.dump({'pool': pool.to_jsonable()}, f, ensure_ascii=False, indent=2)   # {'pool':[...]} = live/materialize 同契约
    log(f"[gp-baseline] deploy bundle → {output};snapshot → {snap}")

    # 跨 run 正交 alpha 库:本轮 collected 内存 pnl 贪心挑 within-run 正交高 IC append(只存因子)
    from rl.alpha_pool import append_run
    lib_path = Path(ini('alpha_library', 'path', 'output/alpha_library.json'))
    n_lib = append_run(collected, lib_path, ini('alpha_library', 'corr_max', 0.5),
                       int(ini('alpha_library', 'max_size', 500)), run_ts)
    log(f"[gp-baseline] alpha library += {n_lib}(within-run 正交)→ {lib_path}")


# ============================================================================
# alphasage — AlphaSAGE GFlowNet alpha 因子挖掘(独立 baseline,与 GP 平级,非改进 GP)
#   数据/切分/因果 universe/1h 降采样/val 池回报口径 = 与 gp-baseline 逐字一致 → 平行对比。
#   GFN(torch)与 eval 同走 [evaluator].device(V100 → cuda,本地 → cpu;单一旋钮)。
# ============================================================================

def cmd_alphasage(output: str) -> None:
    """AlphaSAGE GFlowNet baseline:load panel → GFN P(α)∝R 搜索(train,reward=R_IC+nov_weight·R_NOV)
    → LinearAlphaPool(同 GP 的 IC+正交Δ+OOS holdout 准入门)→ val 池回报(eval_pool_val,同 GP judge)
    → 落 deploy bundle(output JSON)+ snapshot。device 由 [evaluator] 派发(GFN+eval 同走)。"""
    os.environ['ARROW_DEFAULT_MEMORY_POOL'] = 'system'   # Arrow 还内存(同 gp-baseline)
    from config.config import AlphaSAGEConfig
    from search.alphasage.gfn_trainer import train_alphasage
    asc = AlphaSAGEConfig()
    bcfg = BacktestRewardConfig()
    log(f"[alphasage] train={_GP_TRAIN[0]}~{_GP_TRAIN[1]}  val={_GP_VAL[0]}~{_GP_VAL[1]}  (test held out)")
    data = prepare_gp_data(_GP_TRAIN[0], _GP_TRAIN[1], _GP_VAL[0], _GP_VAL[1], asc.parquet_root)
    pool, _ = train_alphasage(
        data['train_panels'], data['train_y'], capacity=asc.pool_capacity,
        hidden_dim=asc.hidden_dim, n_layers=asc.n_layers,
        lr=asc.lr, batch_size=asc.batch_size, n_episodes=asc.n_episodes,
        nov_weight=asc.nov_weight, entropy_coef=asc.entropy_coef,
        entropy_temperature=asc.entropy_temperature,
        r_sa_weight=asc.r_sa_weight, r_sa_k=asc.r_sa_k, r_sa_tau=asc.r_sa_tau,
        logz_lr=asc.logz_lr, seed=42, device=_EVAL_DEVICE, log=log)
    pool_ic, pool_ret = eval_pool_val(pool, data, bcfg)
    log(f"[alphasage] pool_size={pool.size} val pool_ic={pool_ic:+.4f} pool_ret={pool_ret:+.4f}")
    run_ts = os.environ.get('RUN_TS') or time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    snap = Path(f'logs/alphasage_pool_{run_ts}.jsonl')
    snap.parent.mkdir(parents=True, exist_ok=True)
    with open(snap, 'w', encoding='utf-8') as f:
        f.write(json.dumps({'pool': pool.to_jsonable(), 'val_pool_ic': pool_ic,
                            'val_pool_ret': pool_ret}, ensure_ascii=False) + '\n')
    with open(output, 'w', encoding='utf-8') as f:
        json.dump({'pool': pool.to_jsonable()}, f, ensure_ascii=False, indent=2)   # {'pool':[...]} = live/materialize 同契约
    log(f"[alphasage] deploy bundle → {output};snapshot → {snap}")


# ============================================================================
# materialize  — 物化区间 = train_start 月 ~ test_end 月
# ============================================================================

def cmd_materialize(input_json: str, output_dir: str) -> None:
    import pandas as pd

    fm = FactorMiningConfig()
    start_month = '2023-03'
    end_month   = '2026-04'

    with open(input_json, 'r', encoding='utf-8') as f:
        run = json.load(f)
    trees: List[Tuple[str, AlphaTree]] = [
        (f'alpha_{i:03d}', AlphaTree.from_dict(m['tree']))
        for i, m in enumerate(run['pool'])
    ]
    log(f"[materialize] loaded {len(trees)} alphas from {input_json}")

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    months = month_iter(start_month, end_month)
    log(f"[materialize] months: {len(months)}  ({start_month} ~ {end_month})")

    # 只物化 per-alpha 列;池不定权 → 不再产静态 weighted-combo(下游融合走 AFF 因果滚动 lstsq)。
    for mo in months:
        t0 = time.perf_counter()
        y, m = map(int, mo.split('-'))
        m_first = f'{mo}-01'
        m_last  = f'{mo}-{monthrange(y, m)[1]:02d}'
        bundle = load_panel(
            train_start=m_first, train_end=m_last,
            val_start=None, val_end=None,
            parquet_root=fm.parquet_5m_root,
            primary_horizon=0,
        )
        idx = np.where(bundle.train_mask)[0]
        ts = bundle.timestamps[idx]
        S = bundle.S

        out_data = {
            'decision_time': np.repeat(ts, S),
            'symbol':        np.tile(bundle.symbols, len(idx)),
        }
        for name, tree in trees:
            v = eval_tree(tree, bundle.panels, None)
            out_data[name] = v[idx, :].reshape(-1)

        df_out = pd.DataFrame(out_data)
        feat_cols = [c for c in df_out.columns if c not in ('decision_time', 'symbol')]
        finite_any = np.isfinite(df_out[feat_cols].values).any(axis=1)
        df_out = df_out.loc[finite_any]

        out_path = out_root / f'{mo}.parquet'
        df_out.to_parquet(out_path, engine='pyarrow', compression='zstd', index=False)
        log(f"[materialize] {mo}: {len(df_out)} rows -> {out_path.name}  "
              f"({time.perf_counter()-t0:.1f}s)")


# ============================================================================
# 入口
# ============================================================================

def main(argv=None):
    p = argparse.ArgumentParser(prog='quant', description='公式化 alpha 因子挖掘')
    sub = p.add_subparsers(dest='cmd', required=True)

    pg = sub.add_parser('gp-baseline', help='DEAP 强类型 GP 因子挖掘(NSGA-II;产 alpha pool JSON)')
    pg.add_argument('output', help='deploy bundle JSON 路径')

    pas = sub.add_parser('alphasage', help='AlphaSAGE GFlowNet 因子挖掘(独立 baseline;产 alpha pool JSON)')
    pas.add_argument('output', help='deploy bundle JSON 路径')

    pmat = sub.add_parser('materialize', help='把 alpha JSON 物化到月级 parquet')
    pmat.add_argument('input',  help='alpha JSON 路径')
    pmat.add_argument('output', help='输出根目录')

    args = p.parse_args(argv)
    if args.cmd == 'gp-baseline':
        cmd_gp_baseline(args.output)
    elif args.cmd == 'alphasage':
        cmd_alphasage(args.output)
    else:
        cmd_materialize(args.input, args.output)


if __name__ == '__main__':
    main()
