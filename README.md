# SMD-Net

SAM-guided Multi-view Decoding Network for 3D TOF-MRA intracranial aneurysm segmentation.

Built on [nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet). Launch SMD-Net with `MODEL_NAME=smdnet`.

## Requirements

- Python >= 3.9
- PyTorch >= 2.0 (CUDA recommended)
- [segment_anything](https://github.com/facebookresearch/segment-anything) (SAM ViT-B)

```bash
conda create -n smdnet python=3.10 -y
conda activate smdnet
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -e .
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install git+https://github.com/ChaoningZhang/MobileSAM.git timm
```

Place SAM ViT-B weights as:

```text
$nnUNet_raw/sam_vit_b_01ec64.pth
```

Or set `SAM_VIT_B_WEIGHTS=/path/to/sam_vit_b_01ec64.pth`.

For a lighter encoder during debugging:

```bash
export SAM_ENCODER=vit_t
```

## Environment

```bash
export nnUNet_raw=/path/to/nnUNet_raw
export nnUNet_preprocessed=/path/to/nnUNet_preprocessed
export nnUNet_results=/path/to/nnUNet_results
export MODEL_NAME=smdnet
export PYTHONPATH=$PWD:$PYTHONPATH
```

Prepare data in nnU-Net format, then:

```bash
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
```

## Train

Single fold:

```bash
CUDA_VISIBLE_DEVICES=0 python -m nnunetv2.run.run_training 504 3d_lowres 0
```


Training defaults for `MODEL_NAME=smdnet`:


## Inference

```bash
export INPUT_FOLDER=/path/to/imagesTs
export OUTPUT_FOLDER=/path/to/predictions
export MODEL_FOLDER=/path/to/nnUNetTrainer__nnUNetPlans__3d_lowres
bash scripts/predict.sh
```

## Evaluation

Lesion-level detection metrics (IoU matching):

```bash
python scripts/evaluate_detection.py \
  --gt_folder /path/to/labelsTs \
  --pred_folder /path/to/predictions \
  --output_csv /path/to/metrics.csv
```

