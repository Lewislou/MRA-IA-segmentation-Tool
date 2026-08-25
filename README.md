# SMD‑Net

SAM‑guided Multi‑view Decoding Network for 3D TOF‑MRA intracranial aneurysm segmentation.

> 
> Two forms of distribution are provided: **source‑code‑based research version (SMD‑Net)** and **standalone command‑line Windows executable tool (Zjnu_IA_Seg_Tool)**.
> The research version is built on [nnU‑Net v2](https://link.wtturl.cn/?target=https%3A%2F%2Fgithub.com%2FMIC%25E2%2580%2591DKFZ%2FnnUNet&scene=im&aid=582478&lang=zh). Launch SMD‑Net with environment variable `MODEL_NAME=smdnet`.

## 📋 Requirements

### Research Source‑Code Version

- Python >= 3.9
- PyTorch >= 2.0 (CUDA is strongly recommended)
- [segment‑anything](https://link.wtturl.cn/?target=https%3A%2F%2Fgithub.com%2Ffacebookresearch%2Fsegment%25E2%2580%2591anything&scene=im&aid=582478&lang=zh) (SAM ViT‑B)

```
conda create -n smdnet python=3.10 -y
conda activate smdnet
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -e .
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install git+https://github.com/ChaoningZhang/MobileSAM.git timm
```

Place SAM ViT‑B weights at:

```
$nnUNet_raw/sam_vit_b_01ec64.pth
```

Or specify path via environment variable:

```
export SAM_VIT_B_WEIGHTS=/path/to/sam_vit_b_01ec64.pth
```

For lightweight encoder for debugging purpose:

```
export SAM_ENCODER=vit_t
```

### Stand‑alone Windows Executable Tool (Zjnu_IA_Seg_Tool)

- Windows operating system
- Pre‑trained model files located inside `checkpoints` folder, or custom model directory can be specified via argument
- Input images must be `.nii.gz` NIfTI format

## 📦 Download (Windows Executable Tool)

The pre‑compiled software package can be downloaded from Baidu Netdisk:

> 
> Link: [https://pan.baidu.com/s/1Mt](https://link.wtturl.cn/?target=https%3A%2F%2Fpan.baidu.com%2Fs%2F1Mt&scene=im&aid=582478&lang=zh)‑pkrDpcSrNRUppM6j14Q?pwd=76gq
> Password: `76gq`

### Features of Zjnu_IA_Seg_Tool

- Accept single `.nii.gz` file as input
- Support batch prediction for an entire input folder
- CPU / CUDA GPU device selection available
- Support Chinese characters in input, output and model folder paths
- Output segmentation results in standard NIfTI format

### Folder Structure for Executable Release

```
MRA_IA_Segmentation_Tool/
│
├─ MRA IA Segmentation Tool.exe
├─ checkpoints/
│   ├─ dataset.json
│   ├─ plans.json
│   ├─ fold_0/
│   ├─ fold_1/
│   ├─ fold_2/
│   ├─ fold_3/
│   └─ fold_4/
└─ README.md
```

#### Basic usage of Windows command‑line tool

```
"MRA IA Segmentation Tool.exe" -i INPUT_PATH -o OUTPUT_PATH -m MODEL_PATH
```

Force CPU inference:

```
"MRA IA Segmentation Tool.exe" -i INPUT_PATH -o OUTPUT_PATH -m MODEL_PATH -dev cpu
```

Force GPU inference:

```
"MRA IA Segmentation Tool.exe" -i INPUT_PATH -o OUTPUT_PATH -m MODEL_PATH -dev gpu
```

Auto device selection (try GPU first, fall back to CPU):

```
"MRA IA Segmentation Tool.exe" -i INPUT_PATH -o OUTPUT_PATH -m MODEL_PATH -dev auto
```

## ⚙ Environment Variables (Research Source‑Code Version)

```
export nnUNet_raw=/path/to/nnUNet_raw
export nnUNet_preprocessed=/path/to/nnUNet_preprocessed
export nnUNet_results=/path/to/nnUNet_results
export MODEL_NAME=smdnet
export PYTHONPATH=$PWD:$PYTHONPATH
```

Prepare dataset following nnU‑Net data format, then run preprocessing:

```
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
```

## 🚀 Train (Research Source‑Code Version)

Train a single fold:

```
CUDA_VISIBLE_DEVICES=0 python -m nnunetv2.run.run_training 504 3d_lowres 0
```

Training hyper‑parameters will be automatically applied when `MODEL_NAME=smdnet` is set.

## 🔍 Inference

### Source‑code version

```
export INPUT_FOLDER=/path/to/imagesTs
export OUTPUT_FOLDER=/path/to/predictions
export MODEL_FOLDER=/path/to/nnUNetTrainer__nnUNetPlans__3d_lowres
bash scripts/predict.sh
```

> 
> For Windows users, please use the compiled executable tool described above.

## 📊 Evaluation (Lesion‑level detection metrics)

Compute lesion‑level detection metrics with IoU‑based matching strategy:

```
python scripts/evaluate_detection.py \
  --gt_folder /path/to/labelsTs \
  --pred_folder /path/to/predictions \
  --output_csv /path/to/metrics.csv
```

## 📝 Source codes note

> 
> The full training source code for the Zjnu_IA_Seg_Tool compiled binary will be released soon.

## 📜 License

Distributed under the terms of the [BSD‑3‑Clause](https://link.wtturl.cn/?target=http%3A%2F%2Fopensource.org%2Flicenses%2FBSD%25E2%2580%25913%25E2%2580%2591Clause&scene=im&aid=582478&lang=zh) license.
SMD‑Net / Zjnu_IA_Seg_Tool is free and open‑source software.

## 🐛 Issues

If you encounter any problems, please [file an issue] with detailed reproduction steps and environment information.
