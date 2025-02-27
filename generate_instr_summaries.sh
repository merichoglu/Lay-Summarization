#!/bin/bash
#SBATCH --job-name=generate_instr_summaries
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --gpus=v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=18:00:00
#SBATCH -A cis230089p
#SBATCH --output=generate_instr_output_with_new_prompt.log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=meric.haliloglu@metu.edu.tr

# Load necessary modules
module purge
module load anaconda3
module load cuda  # Load CUDA explicitly!

# Debug: Check if CUDA is available before running Python
echo "Checking CUDA availability before running Python..."
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"

# Enable PyTorch CUDA memory optimizations
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_LAUNCH_BLOCKING=1

# Run the instruction inference script with required arguments
python -u generate_instr_summaries.py \
  --config_path=inference_config.json

