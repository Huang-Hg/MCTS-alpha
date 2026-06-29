#!/usr/bin/env bash
# 编译 alpha 算子 Python C 扩展。
#   Linux 原生 → _ops.<extension-suffix>.so
#   WSL → Windows .pyd(用 msys2 ucrt64 gcc)
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

# WSL 检测:跟 backtest/build.sh 同样逻辑,grep /proc/version 判 microsoft/wsl
IS_WSL=0
if grep -qiE 'microsoft|wsl' /proc/version 2>/dev/null; then
    IS_WSL=1
fi

if [ "$(uname -s)" = "Linux" ] && [ "$IS_WSL" = "0" ]; then
    PY="${PYTHON:-python3}"
    EXT_SUFFIX="$($PY -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
    PY_INC="$($PY -c 'import sysconfig; print(sysconfig.get_paths()["include"])')"
    NUMPY_INC="$($PY -c 'import numpy; print(numpy.get_include())')"
    OUT="$HERE/_ops${EXT_SUFFIX}"

    gcc \
        -O3 -march=native -ffast-math -fno-finite-math-only -fno-strict-aliasing \
        -fopenmp -fPIC \
        -Wall -Wextra -Wno-unused-parameter \
        -shared \
        -I"$PY_INC" -I"$NUMPY_INC" -I"$HERE" \
        "$HERE/ops_elementwise.c" "$HERE/ops_ts.c" "$HERE/ops_cs.c" "$HERE/ops_metrics.c" "$HERE/_ops_module.c" \
        -o "$OUT" \
        -lgomp -lm
else
    GCC="${GCC:-/mnt/c/msys64/ucrt64/bin/gcc.exe}"
    PY_PREFIX_WIN="${PY_PREFIX_WIN:-C:/Users/WIT_User/AppData/Local/Programs/Python/Python313}"
    NUMPY_INC_WIN="${NUMPY_INC_WIN:-D:/03learn/machine/.venv/Lib/site-packages/numpy/_core/include}"
    PY_INC="$PY_PREFIX_WIN/include"
    PY_LIB="$PY_PREFIX_WIN/libs"
    OUT="$(wslpath -w "$HERE/_ops.cp313-win_amd64.pyd")"

    "$GCC" \
        -O3 -march=native -ffast-math -fno-finite-math-only -fno-strict-aliasing \
        -fopenmp \
        -static -static-libgcc \
        -Wall -Wextra -Wno-unused-parameter \
        -shared \
        -I"$PY_INC" -I"$NUMPY_INC_WIN" -I"$(wslpath -w "$HERE")" \
        "$(wslpath -w "$HERE/ops_elementwise.c")" "$(wslpath -w "$HERE/ops_ts.c")" "$(wslpath -w "$HERE/ops_cs.c")" "$(wslpath -w "$HERE/ops_metrics.c")" "$(wslpath -w "$HERE/_ops_module.c")" \
        -L"$PY_LIB" -lpython313 \
        -o "$OUT" \
        -lm
    OUT="$HERE/_ops.cp313-win_amd64.pyd"
fi

echo "Built $OUT ($(stat -c '%s' "$OUT") bytes)"
