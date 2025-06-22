#!/bin/bash
#SBATCH --job-name=stable-audio-train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=12:00:00

set -e

# export SLURM_CONF=/home/sladmitet/slurm/slurm.conf # put this in .bashrc.user
export PATH="$HOME/mambaforge/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate /itet-stor/feigao/net_scratch/fg-envs/vec-env

export WANDB_API_KEY=
export WANDB_MODE=online

cd /itet-stor/feigao/home/stable-audio-tools-fsq

# Check if arguments are provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <wandb_project> <wandb_run_name>"
    exit 1
fi

WANDB_PROJECT=$1
WANDB_RUN_NAME=$2

srun bash train_server.sh "$WANDB_PROJECT" "$WANDB_RUN_NAME"