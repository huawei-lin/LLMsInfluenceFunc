#! /bin/bash

set -x
conda activate llama
shell_dir=$(cd "$(dirname "$0")";pwd)
echo "shell_dir: ${shell_dir}"

python ../MP_main.py --config='./config.json' | tee infl_output.txt

