export CUDA_VISIBLE_DEVICES=0

# GOLA-B
python ../profile_model.py GOLA dinov2 --device cuda

# GOLA-L
#python ../profile_model.py GOLA dinov2 --mixin_config large --device cuda
