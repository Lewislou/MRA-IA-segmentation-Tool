# Zjnu_IA_Seg_Tool

MRA IA Segmentation Tool is a command-line application for intracranial aneurysm segmentation from MRA images.

## Features

- Supports single `.nii.gz` file input
- Supports batch prediction from a folder
- Supports CPU and CUDA device selection
- Automatically handles Chinese paths for input, output, and model folders
- Saves segmentation results in NIfTI format

## Requirements

- Windows
- Model files in the `checkpoints` folder, or specify a custom model folder
- Input images in `.nii.gz` format

## Folder Structure

Example:

```text
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

## Basic usage
```shell
"MRA IA Segmentation Tool.exe" -i INPUT_PATH -o OUTPUT_PATH -m MODEL_PATH
```

## Force CPU
```shell
"MRA IA Segmentation Tool.exe" -i INPUT_PATH -o OUTPUT_PATH -m MODEL_PATH -dev cpu
```

## Force GPU
```shell
"MRA IA Segmentation Tool.exe" -i INPUT_PATH -o OUTPUT_PATH -m MODEL_PATH -dev gpu
```
## Auto selection
```shell
"MRA IA Segmentation Tool.exe" -i INPUT_PATH -o OUTPUT_PATH -m MODEL_PATH -dev auto
```

## Source codes and training
The source codes of this model and the training process will be released soon.

## License

Distributed under the terms of the [BSD-3](http://opensource.org/licenses/BSD-3-Clause) license,"Zjnu_IA_Seg_Tool" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

