#!/usr/bin/env bash
# 统一构建 backtest CPU C/C++ 扩展(OS 分支只此一份,替代旧 backtest/build.sh + ops/build.sh):
#   ops    (C)   → ops/_ops.<ext>      因子算子(backtest.ops._ops)
#   engine (C++) → _bt.<ext>           policy 抽象组合回测(backtest._bt)
#   Linux 原生 → .so;WSL → Windows .pyd(msys2 ucrt64 g++;venv 是 Windows 二进制)。
# 通用:两分支均**从目标解释器派生** ext-suffix / include / numpy / lib(无硬编码 py 版本/路径);
# env 旋钮 PYTHON(目标解释器,Linux 默认 python3、WSL 默认项目 venv)、GXX(WSL 编译器)。
# CUDA(ops_cuda)走 cupy NVRTC **运行时编译**,无 .so/.pyd 产物 → 仍由 ops_cuda/build.sh
# 触发 import-verify(非 OS-branch 编译重复,故不并入本脚本)。
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

IS_WSL=0
grep -qiE 'microsoft|wsl' /proc/version 2>/dev/null && IS_WSL=1

OPT="-O3 -march=native -ffast-math -fno-finite-math-only -fno-strict-aliasing -fopenmp"
WARN="-Wall -Wextra -Wno-unused-parameter"
OPS_SRC="$HERE/ops/ops_ts.cpp $HERE/ops/ops_xs.cpp $HERE/ops/_ops_module.cpp"
ENG_SRC="$HERE/engine/engine.cpp $HERE/engine/_bt_module.cpp"

if [ "$(uname -s)" = "Linux" ] && [ "$IS_WSL" = "0" ]; then
    PY="${PYTHON:-python3}"
    EXT="$($PY -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
    PYI="$($PY -c 'import sysconfig; print(sysconfig.get_paths()["include"])')"
    NPI="$($PY -c 'import numpy; print(numpy.get_include())')"
    INC="-I$PYI -I$NPI -I$HERE"
    BASE="$OPT -fPIC $WARN -shared"
    g++ -std=c++17 $BASE "-I$HERE/ops"    $INC $OPS_SRC -o "$HERE/ops/_ops${EXT}" -lgomp -lm
    g++ -std=c++17 $BASE "-I$HERE/engine" $INC $ENG_SRC -o "$HERE/_bt${EXT}"       -lgomp -lm
    echo "Built $HERE/ops/_ops${EXT} + $HERE/_bt${EXT}"
else
    # WSL:交叉构建 Windows .pyd —— Windows venv python.exe(WSL py 缺依赖/扩展)+ msys2 ucrt64 g++。
    # 版本/路径**全从目标解释器派生**(不硬编码 py 版本 / 用户名 / numpy·base 路径);仅 PYTHON
    # (目标解释器)、GXX(编译器)两个 env 旋钮。sysconfig 返 Windows 反斜杠 → 转正斜杠喂 gcc.exe。
    GXX="${GXX:-/mnt/c/msys64/ucrt64/bin/g++.exe}"
    PYWIN="${PYTHON:-/mnt/d/03learn/machine/.venv/Scripts/python.exe}"
    q() { "$PYWIN" -c "$1" | tr -d '\r' | tr '\\' '/'; }
    EXT="$(q 'import sysconfig;print(sysconfig.get_config_var("EXT_SUFFIX"))')"            # .cp3XX-win_amd64.pyd
    PYINC="$(q 'import sysconfig;print(sysconfig.get_paths()["include"])')"                 # Python.h(base 安装)
    NPINC="$(q 'import numpy;print(numpy.get_include())')"                                  # venv numpy C API
    LIBDIR="$(q 'import sys;print(sys.base_prefix)')/libs"                                  # python3XX.lib 所在
    PYLIB="$(q 'import sysconfig;print("python"+sysconfig.get_config_var("py_version_nodot"))')"  # python3XX
    INC="-I$PYINC -I$NPINC -I$(wslpath -w "$HERE")"
    BASE="$OPT -static -static-libgcc $WARN -shared"
    wp() { for a in "$@"; do wslpath -w "$a"; done; }
    "$GXX" -std=c++17 $BASE -static-libstdc++ "-I$(wslpath -w "$HERE/ops")"    $INC $(wp $OPS_SRC) \
        -L"$LIBDIR" -l"$PYLIB" -o "$(wslpath -w "$HERE/ops/_ops$EXT")" -lm
    "$GXX" -std=c++17 $BASE -static-libstdc++ "-I$(wslpath -w "$HERE/engine")" $INC $(wp $ENG_SRC) \
        -L"$LIBDIR" -l"$PYLIB" -o "$(wslpath -w "$HERE/_bt$EXT")" -lm
    echo "Built $HERE/ops/_ops$EXT + $HERE/_bt$EXT  (PYTHON=$PYWIN EXT=$EXT)"
fi
