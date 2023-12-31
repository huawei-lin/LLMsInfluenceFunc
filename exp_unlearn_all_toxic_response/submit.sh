#!/bin/bash -l
# NOTE the -l flag!
#
# The name of your job
#SBATCH --job-name=test_55
#
# Where to save the output and error messages for your job
# %x will fill in your job name, %j will fill in your job ID
#SBATCH --output=output
#SBATCH --error=error
#
# Compute resources
#SBATCH --ntasks=1
#SBATCH --time=20:00:00

#SBATCH --cpus-per-task=4         # CPU cores/threads
#SBATCH --mem=80000mb
#SBATCH --gpus-per-node=a100:1
#
# Your slurm account
#SBATCH --account=sssstrees
#
# The partition to run your job on
#SBATCH --partition=tier3

# The code you actually need to run goes here


nvidia-smi
python ../../MP_main.py --config='./config.json' | tee std.txt
