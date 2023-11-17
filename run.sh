sleep 45m

cp /data/home/wufeim/dst_cvpr/dst_omni3d/DST_arkitscenes_aug_val.json /fsx/wufeim/omni3d/datasets/Omni3D
cp /data/home/wufeim/dst_cvpr/dst_omni3d/DST_arkitscenes_aug_train.json /fsx/wufeim/omni3d/datasets/Omni3D

sbatch omni3d_arkit_dst3d_aug_025.slurm
