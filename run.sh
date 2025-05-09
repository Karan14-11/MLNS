#!/bin/bash
#SBATCH --ntasks=1            # Run a single task
#SBATCH --cpus-per-task=10    # Number of CPU cores per task
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=3000    # Memory per CPU in MB
#SBATCH --mail-type=ALL       # Send email notifications for all events
#SBATCH -w gnode079        # Specify the compute node
#SBATCH -o ./DeepTGIN/output.out     # Standard output and error log

# Initialize Conda and activate environment



cd DeepTGIN
python3 main_gcn.py