#!/usr/bin/env bash
# backtest/ops_cuda 的「构建+验证」—— CUDA 算子经 cupy NVRTC **运行时编译**(无 nvcc、无
# 预编译 .so/.pyd 产物),故此脚本职责 = 触发 NVRTC 编译(import 即编 kernels.cu 全实例)+ 逐算子
# 对真 C 验证。与 CPU 侧 backtest/ops/build.sh(编 .so/.pyd)对称但形态不同。
#
# PYTHON 须带 cupy + NVIDIA GPU:WSL/本地用 tmp/cuda_toy_venv,server 用各自 venv(见 reference 记忆)。
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PY="${PYTHON:-$ROOT/tmp/cuda_toy_venv/bin/python}"

echo "[1/2] NVRTC 编译 kernels.cu(import backtest.ops_cuda → 编 f64/f32 全实例)..."
cd "$ROOT"
"$PY" -c "import backtest.ops_cuda as g; print('  OK dtype=%s FP64:FP32=1:%d optin=%dKB' % (__import__('numpy').dtype(g.DTYPE).name, g._FP64_RATIO, g._OPTIN//1024))"

echo "[2/2] 逐算子 vs 真 C 验证(需 tmp/ops_build/_ops,缺则先 bash scripts/build_ops_so.sh)..."
"$PY" scripts/smoke_ops_cuda.py | tail -3
echo "Done."
