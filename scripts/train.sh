#!/usr/bin/env bash
set -euo pipefail

DATASET_DIR="data/generated_dataset"

if [[ ! -f "${DATASET_DIR}/seeds.json" ]]; then
	echo "[INFO] seeds.json not found, preparing dataset index..."
	if command -v conda >/dev/null 2>&1; then
		conda run -n ip2p python dataset_creation/prepare_dataset.py "${DATASET_DIR}"
	else
		/home/user/anaconda3/envs/ip2p/bin/python dataset_creation/prepare_dataset.py "${DATASET_DIR}"
	fi
fi

if command -v conda >/dev/null 2>&1; then
	conda run -n ip2p python main.py --name ip2p-small --base configs/train_small.yaml --train --gpus 1
else
	/home/user/anaconda3/envs/ip2p/bin/python main.py --name ip2p-small --base configs/train_small.yaml --train --gpus 1
fi