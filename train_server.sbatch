#!/bin/bash
#SBATCH --job-name=stable-audio-train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --gres=gpu:6
#SBATCH --nodelist=tikgpu06
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -e
# SBATCH --nodelist=tikgpu05
# SBATCH --constraint=RTX3090|A100|RTXA6000
# export SLURM_CONF=/home/sladmitet/slurm/slurm.conf # put this in .bashrc.user
export PATH="$HOME/mambaforge/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate /itet-stor/feigao/net_scratch/fg-envs/vec-env

export WANDB_API_KEY=81c0145df67da30b28a589b1a1b66eb117859969
export WANDB_MODE=online

cd /itet-stor/feigao/home/stable-audio-tools

# Check if arguments are provided
if [ $# -ne 5 ]; then
    echo "Usage: $0 <config_name> <lambda> <wandb_project> <wandb_run_name> <continue|new>"
    exit 1
fi

CONFIG_NAME=$1
LAMBDA=$2
WANDB_PROJECT=$3
WANDB_RUN_NAME=$4
RESUME_MODE=$5

srun bash train_server.sh "$CONFIG_NAME" "$LAMBDA" "$WANDB_PROJECT" "$WANDB_RUN_NAME" "$RESUME_MODE"