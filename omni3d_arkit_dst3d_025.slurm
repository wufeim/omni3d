#!/bin/bash
#
#SBATCH --job-name=omni3d_arkit_dst3d_025
#SBATCH --error=/fsx/wufeim/omni3d/exp/omni3d_arkit_dst3d_025/%j_0_log.err
#SBATCH --output=/fsx/wufeim/omni3d/exp/omni3d_arkit_dst3d_025/%j_0_log.out
#SBATCH --gpus-per-node=8
#SBATCH --nodes=1
#SBATCH --partition=learnai4rl
#SBATCH --cpus-per-task=10
#SBATCH --time=05-00:00:00
#SBATCH --mem=320GB
#SBATCH --ntasks-per-node=8
#SBATCH --open-mode=append

/data/home/wufeim/miniconda3/envs/omni3d/bin/python3 tools/train_net.py --config-file configs/Base_DST3D_in_ARKitScenes_025.yaml --num-gpus 8 --resume
