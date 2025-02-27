#!/bin/bash
#SBATCH --job-name=instr_finetune_job
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --gpus=v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH -A cis230089p
#SBATCH --output=instr_finetune_output_with_new_prompt.log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=meric.haliloglu@metu.edu.tr

# Load necessary modules
module purge
module load anaconda3

# Enable PyTorch CUDA memory optimizations
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_LAUNCH_BLOCKING=1

# Run the instruction finetuning script
python instruction_finetune.py --config_path=instruction_finetune_config.json
