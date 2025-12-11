export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export NCCL_P2P_DISABLE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

GPU_NUM=$(echo $CUDA_VISIBLE_DEVICES | tr -cd ',' | wc -c)
((GPU_NUM = GPU_NUM + 1))

gpustat

output_dir="../output/test"
weight_path="../output/gola_b224.bin"
timestamp=$(date +"%Y.%m.%d-%H.%M.%S")

mkdir -p $output_dir

# GOLA-B
python ../main.py GOLA dinov2 --eval --mixin_config evaluation --distributed_nproc_per_node "${GPU_NUM}" --distributed_do_spawn_workers --weight_path $weight_path --device cuda --disable_wandb --output_dir=$output_dir |& tee -a "$output_dir/eval_stdout-$timestamp.log"

# GOLA-L
#python ../main.py GOLA dinov2 --eval --mixin_config large --mixin_config evaluation --distributed_nproc_per_node "${GPU_NUM}" --distributed_do_spawn_workers --weight_path $weight_path --device cuda --disable_wandb --output_dir=$output_dir |& tee -a "$output_dir/eval_stdout-$timestamp.log"