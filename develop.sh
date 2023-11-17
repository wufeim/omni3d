python3 run_with_submitit.py \
    --job_name omni3d_arkit_baseline_05 \
    --job_dir exp/omni3d_arkit_baseline_05 \
    --partition learnai4rl \
    --timeout 72 \
    --nodes 1 \
    --ngpus 8 \
    --config-file configs/Base_Omni3D_in_ARKitScenes_05.yaml \
    --num-gpus 8

exit

python train_net.py \
    --config-file configs/Base_Omni3D_in_ARKitScenes.yaml \
    --num-gpus 8

exit

python tools/train_net.py \
    --config-file configs/Base_Omni3D_in_ARKitScenes.yaml \
    --num-gpus 8 \
    OUTPUT_DIR /data/home/wufeim/dst_cvpr/omni3d/exp/omni3d_8gpu_ARKitScenes

exit
