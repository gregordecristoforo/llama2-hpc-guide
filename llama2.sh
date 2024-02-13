#!/bin/bash

#SBATCH --job-name Llama2   ## Name of the job
#SBATCH --output slurm-%j.out   ## Name of the output-script (%j will be replaced with job number)
#SBATCH --account nn9997k   ## The billed account
#SBATCH --time=00:15:00   ## Walltime of the job
#SBATCH --partition=a100  ## Selected partition
#SBATCH --mem-per-cpu=32000 ## Memory allocated to each task
#SBATCH --ntasks=1   ## Number of tasks that will be allocated
#SBATCH --gpus=1   ## CPUs allocated per task

set -o errexit   ## Exit the script on any error
set -o nounset   ## Treat any unset variables as an error

# necessary for A100 GPUS
module --force swap StdEnv Zen2Env
module load Python/3.10.8-GCCcore-12.2.0

source llama2/bin/activate

python llama2.py
