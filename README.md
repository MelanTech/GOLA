# [AAAI2026] Group Orthogonal Low-Rank Adaptation for RGB-T Tracking

Official implementation of [Group Orthogonal Low-Rank Adaptation for RGB-T Tracking](https://arxiv.org/abs/2512.05359). (AAAI 2026)

[[Pretrained Models](https://drive.google.com/drive/folders/1p9DDPcP251-_mvLVUQ8BXbdEjDA4sttM?usp=sharing)] [[Weights](https://drive.google.com/drive/folders/1WQ7bZ9ZmXkKTryShMsTdmG5h1yLwGGxz?usp=sharing)] [[Raw Results](https://drive.google.com/drive/folders/1tScEdLUNNtSaGqXuchCM4I5LC8K6S7jX?usp=sharing)] [[Training Logs](https://drive.google.com/drive/folders/1gaN1ck4n75goq8WRIPWr3ECra0cbjw8i?usp=sharing)]

<img src="./assets/framework.jpg"/>

>**Abstract:** Parameter-efficient fine-tuning has emerged as a promising paradigm in RGB-T tracking, enabling downstream task adaptation by freezing pretrained parameters and fine-tuning only a small set of parameters. This set forms a rank space made up of multiple individual ranks, whose expressiveness directly shapes the model's adaptability. However, quantitative analysis reveals low-rank adaptation exhibits significant redundancy in the rank space, with many ranks contributing almost no practical information. This hinders the model's ability to learn more diverse knowledge to address the various challenges in RGB-T tracking. To address this issue, we propose the Group Orthogonal Low-Rank Adaptation (GOLA) framework for RGB-T tracking, which effectively leverages the rank space through structured parameter learning. Specifically, we adopt a rank decomposition partitioning strategy utilizing singular value decomposition to quantify rank importance, freeze crucial ranks to preserve the pretrained priors, and cluster the redundant ranks into groups to prepare for subsequent orthogonal constraints. We further design an inter-group orthogonal constraint strategy. This constraint enforces orthogonality between rank groups, compelling them to learn complementary features that target diverse challenges, thereby alleviating information redundancy. Experimental results demonstrate that GOLA effectively reduces parameter redundancy and enhances feature representation capabilities, significantly outperforming state-of-the-art methods across four benchmark datasets and validating its effectiveness in RGB-T tracking tasks.

## News
**[Dec. 24, 2025]**
* We’re thrilled to release [MMLoRAT](https://github.com/MelanTech/MMLoRAT), a multimodal extension of LoRAT with improved conciseness and overall refinement.

**[Dec. 11, 2025]**
* We have released the code, weights, raw results and training logs.

**[Nov. 08, 2025]**
* Our GOLA has been accepted by AAAI 2026.

## Performance
<table style="text-align: center;">
  <tr>
    <th rowspan="2">Variant</th>
    <th colspan="3">LasHeR</th>
    <th colspan="2">RGBT234</th>
    <th colspan="2">RGBT210</th>
    <th colspan="2">GTOT</th>
    <th rowspan="2">FPS</th>
  </tr>
  <tr>
    <th>PR(%)</th>
    <th>NPR(%)</th>
    <th>SR(%)</th>
    <th>MPR(%)</th>
    <th>MSR(%)</th>
    <th>PR(%)</th>
    <th>SR(%)</th>
    <th>MPR(%)</th>
    <th>MSR(%)</th>
  </tr>
  <tr>
    <td>GOLA-B</td>
    <td>77.5</td>
    <td>73.9</td>
    <td>61.6</td>
    <td>92.2</td>
    <td>69.5</td>
    <td>90.9</td>
    <td>67.0</td>
    <td>92.8</td>
    <td>78.5</td>
    <td>125</td>
  </tr>
  <tr>
    <td>GOLA-L</td>
    <td>78.1</td>
    <td>74.5</td>
    <td>61.9</td>
    <td>92.8</td>
    <td>71.3</td>
    <td>92.0</td>
    <td>68.7</td>
    <td>95.3</td>
    <td>80.9</td>
    <td>64</td>
  </tr>
</table>

## Prerequisites
### Environment
Assuming you have a `Python 3.10.15` environment with pip installed.

#### system packages (ubuntu)
```shell
apt update
apt install -y libturbojpeg
```
#### install pytorch
```shell
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
```

#### extra python packages
```shell
pip install -r requirements.txt
```
This codebase should also work on Windows and macOS for debugging purposes.

### Dataset
The paths should be organized as follows:
```
-- LasHeR/trainingset
    |-- 1boygo
    |-- 1handsth
    ...
```

### Prepare ```consts.yaml```
Fill in the paths.
```yaml
LasHeR_PATH: '/path/to/LasHeR0428/'
RGBT234_PATH: '/path/to/RGBT234/'
RGBT210_PATH: '/path/to/RGBT210/'
GTOT_PATH: '/path/to/GTOT/'
```

## Quick Start
* Our code performs evaluation automatically when model training is complete. 
  * **Model weight** is saved in ```/path/to/output/run_id/checkpoint/epoch_{last}/model.bin```.
  * **Performance metrics** can be found on terminal output.
  * **Tracking results** are saved in ```/path/to/output/run_id/eval/epoch_{last}/```.

* The performance metrics obtained from automatic evaluation **differ from** official evaluation tools, so we recommend using official tools for re-evaluation.

### Preparation for pretrained models
* Download [base.bin & large.bin](https://drive.google.com/drive/folders/1p9DDPcP251-_mvLVUQ8BXbdEjDA4sttM?usp=sharing) and put them in the `pretrained_models` folder.
* Execute `pretrained_models/params_svd_grouping.py` to group the ranks of LoRA parameters.
```shell
python params_svd_grouping.py --weight ./base.bin --n_retain 16 --n_groups 8
python params_svd_grouping.py --weight ./large.bin --n_retain 16 --n_groups 8
```
* New weights will be saved in the same directory as the original weights.
* You can skip this step and directly download `base_gola_r16_g8.bin` and `large_gola_r16_g8.bin`.

### Training
* Using `sh/train.sh` for training (Linux with NVIDIA GPU only)
```shell
export CUDA_VISIBLE_DEVICES=0,1  # Specify your GPU IDs

output_dir="/path/to/output/directory"  # Specify your output directory
weight_path="/path/to/pretrained/weight"  # Specify your pretrained model path

# Dry run
python ../main.py GOLA dinov2 --dry_run --distributed_nproc_per_node "${GPU_NUM}" --distributed_do_spawn_workers --disable_wandb --weight_path $weight_path

# GOLA-B
python ../main.py GOLA dinov2 --distributed_nproc_per_node "${GPU_NUM}" --distributed_do_spawn_workers --disable_wandb --weight_path $weight_path --output_dir="$output_dir" |& tee -a "$output_dir/train_stdout-$timestamp.log"

# GOLA-L
python ../main.py GOLA dinov2 --mixin_config large --distributed_nproc_per_node "${GPU_NUM}" --distributed_do_spawn_workers --disable_wandb --weight_path $weight_path --output_dir="$output_dir" |& tee -a "$output_dir/train_stdout-$timestamp.log"
```

### Evaluation
* We use [rgbt](https://github.com/opacity-black/RGBT_toolkit) library to evaluate the performance on RGB-T datasets.
* You can skip steps 1 and 2 and simply use the zip file saved during the training phase.

You can run evaluation with the following procedure to reproduce the results reported in the paper:

1. Edit ```config/_dataset/test-mm.yaml``` to specify the dataset to be evaluated.
```yaml
datasets:
  - name: "LasHeR"
    type: "MMOT"
    splits:
      - "test"

#  - name: "RGBT234"  # Uncomment to evaluate RGBT234
#    type: "MMOT"

#  - name: "RGBT210"  # Uncomment to evaluate RGBT210
#    type: "MMOT"

#  - name: "GTOT"  # Uncomment to evaluate GTOT
#    type: "MMOT"
```
* Note: Uncomment one dataset at a time to evaluate.

2. Edit ```sh/test.sh``` to specify the weight path and output directory.
```shell
output_dir="/path/to/output"
weight_path="path/to/weight.bin"
timestamp=$(date +"%Y.%m.%d-%H.%M.%S")

mkdir -p $output_dir

# Evaluation with GOLA-B
python ../main.py GOLA dinov2 --eval --mixin_config evaluation --distributed_nproc_per_node "${GPU_NUM}" --distributed_do_spawn_workers --weight_path $weight_path --device cuda --disable_wandb --output_dir=$output_dir |& tee -a "$output_dir/eval_stdout-$timestamp.log"

# Evaluation with GOLA-L
python ../main.py GOLA dinov2 --eval --mixin_config evaluation --mixin_config large --distributed_nproc_per_node "${GPU_NUM}" --distributed_do_spawn_workers --weight_path $weight_path --device cuda --disable_wandb --output_dir=$output_dir |& tee -a "$output_dir/eval_stdout-$timestamp.log"
```

3. Unzip the tracking results to a folder of your choice.

4. Edit and run the evaluation script in ```sh/evaluation.sh```.
```shell
# Evaluate on LasHeR
python ../evaluation.py lasher --tracker_names GOLA --result_paths /path/to/tracking/results

# Evaluate on RGBT234
python ../evaluation.py rgbt234 --tracker_names GOLA --result_paths /path/to/tracking/results

# Evaluate on RGBT210
python ../evaluation.py rgbt210 --tracker_names GOLA --result_paths /path/to/tracking/results

# Evaluate on GTOT
python ../evaluation.py gtot --tracker_names GOLA --result_paths /path/to/tracking/results
```

### Profile Model
* Using `profile_model.py` for model profiling.
```shell
python ../profile_model.py GOLA dinov2 --device cuda  # GOLA-B
python ../profile_model.py GOLA dinov2 --mixin_config large --device cuda  # GOLA-L
```

## Acknowledgements
- This repo is based on [LoRAT](https://github.com/LitingLin/LoRAT), we thank for it's `trackit` framework, which helps us to quickly implement our ideas.
- We thank the [rgbt](https://github.com/opacity-black/RGBT_toolkit) library for facilitating evaluation in a Python environment.

## Citation
```bibtex
@inproceedings{gola,
  title={Group Orthogonal Low-Rank Adaptation for RGB-T Tracking},
  author={Shao, Zekai and Hu, Yufan and Liu, jingyuan and Fan, Bin and Liu, Hongmin},
  booktitle={AAAI},
  year={2026}
} 
```