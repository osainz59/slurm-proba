#!/bin/bash
#SBATCH --job-name=MLM_prompt
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1-00:00:00
#SBATCH --mem=1GB
#SBATCH --gres=gpu:1
#SBATCH --output=.slurm/MLM_prompt.log
#SBATCH --error=.slurm/MLM_prompt.err

# Activamos el entorno virtual que necesitemos
source /var/python3envs/transformers-4.6.0/bin/activate
export TRANSFORMERS_CACHE="/gaueko0/transformers_cache/"

# # LLamamos a nuestro script de python
echo "Niri sagarrak gustatzen [MASK] !" | srun python mlm_prompt.py --topk 10
# srun python mlm_prompt.py

# Simulacion de un entrenamiento
# srun python training_simulation.py