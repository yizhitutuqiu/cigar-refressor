#!/usr/bin/env bash
set -euo pipefail

usage() {
	echo "用法: $0 <训练运行目录|tb目录> [起始端口(默认6006)]"
	echo "例如: $0 /data/litengmo/ml-test/cifar_regressor/checkpoints/hier_vit_small/20251105_113716 6006"
	echo "或:    $0 /data/litengmo/ml-test/cifar_regressor/checkpoints/hier_vit_small/20251105_113716/tb 6006"
}

if [[ $# -lt 1 ]]; then
	usage
	exit 1
fi

RUN_DIR="$1"
BASE_PORT="${2:-6006}"

# 若传入的是 run 子目录名（不含全路径），尝试在项目 checkpoints 下匹配
if [[ ! -d "$RUN_DIR" ]] && [[ -d "/data/litengmo/ml-test/cifar_regressor/checkpoints/$RUN_DIR" ]]; then
	RUN_DIR="/data/litengmo/ml-test/cifar_regressor/checkpoints/$RUN_DIR"
fi

if [[ ! -d "$RUN_DIR" ]]; then
	echo "错误: 目录不存在: $RUN_DIR" >&2
	exit 1
fi

TB_DIR="$RUN_DIR"
if [[ -d "$RUN_DIR/tb" ]]; then
	TB_DIR="$RUN_DIR/tb"
elif [[ "$(basename "$RUN_DIR")" != "tb" ]]; then
	echo "错误: 未找到 TensorBoard 日志目录：$RUN_DIR 或 $RUN_DIR/tb" >&2
	exit 1
fi

if ! command -v tensorboard >/dev/null 2>&1; then
	echo "错误: 未找到 tensorboard 命令，请先安装 (pip install tensorboard)" >&2
	exit 1
fi

is_port_free() {
	local p="$1"
	python3 - "$p" >/dev/null 2>&1 <<'PY'
import socket, sys
p = int(sys.argv[1])
s = socket.socket()
try:
	s.bind(('0.0.0.0', p))
	s.close()
	sys.exit(0)
except OSError:
	sys.exit(1)
PY
}

find_free_port() {
	local start="$1"
	local end="$2"
	local p="$start"
	while [[ "$p" -le "$end" ]]; do
		if is_port_free "$p"; then
			echo "$p"
			return 0
		fi
		p=$((p+1))
	done
	return 1
}

# 在 [BASE_PORT, BASE_PORT+100] 范围内寻找可用端口
PORT="$(find_free_port "$BASE_PORT" $((BASE_PORT+100)) || true)"
if [[ -z "$PORT" ]]; then
	echo "错误: 在端口区间 [$BASE_PORT, $((BASE_PORT+100))] 内未找到可用端口" >&2
	exit 1
fi

echo "启动 TensorBoard: logdir=$TB_DIR, host=0.0.0.0, port=$PORT"
echo "按 Ctrl-C 结束。"
tensorboard --logdir "$TB_DIR" --host 0.0.0.0 --port "$PORT"


