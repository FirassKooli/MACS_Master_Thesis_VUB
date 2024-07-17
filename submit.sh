#!/bin/bash
#SBATCH --job-name=
#SBATCH --time=01:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=5

module load matplotlib/3.5.2-foss-2022a
module load TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0
module load wandb/0.13.4-GCCcore-11.3.0
module load scikit-learn/1.1.2-foss-2022a
module load Seaborn/0.12.1-foss-2022a

# Execute the Python script and redirect both stdout and stderr to an output log file
python HYDRA_Script_7_Muscle_Identification.py > output.log 2>&1