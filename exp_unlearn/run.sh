#!/bin/bash

export CUDA_VISIBLE_DEVICES="0,1"

master_port=`shuf -i 12000-30000 -n 1`

cd ..
torchrun --nproc_per_node=2 --master_port=${master_port} unlearn.py
