export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export NCCL_P2P_DISABLE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

GPU_NUM=$(echo $CUDA_VISIBLE_DEVICES | tr -cd ',' | wc -c)
((GPU_NUM = GPU_NUM + 1))

output_dir="../output/"
weight_path="../pretrained_models/base_gola_r16_g8.bin"
timestamp=$(date +"%Y.%m.%d-%H.%M.%S")

mkdir -p "$output_dir"

#dry-run
#python ../main.py GOLA dinov2 --dry_run --distributed_nproc_per_node "${GPU_NUM}" --distributed_do_spawn_workers --disable_wandb --weight_path $weight_path

# GOLA-B
python ../main.py GOLA dinov2 --distributed_nproc_per_node "${GPU_NUM}" --distributed_do_spawn_workers --disable_wandb --weight_path $weight_path --output_dir="$output_dir" |& tee -a "$output_dir/train_stdout-$timestamp.log"

# GOLA-L
#python ../main.py GOLA dinov2 --mixin_config large --distributed_nproc_per_node "${GPU_NUM}" --distributed_do_spawn_workers --disable_wandb --weight_path $weight_path --output_dir="$output_dir" |& tee -a "$output_dir/train_stdout-$timestamp.log"