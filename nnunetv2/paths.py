import os

nnUNet_raw = os.environ.get("nnUNet_raw")
nnUNet_preprocessed = os.environ.get("nnUNet_preprocessed")
nnUNet_results = os.environ.get("nnUNet_results")

if nnUNet_results is not None:
    os.makedirs(nnUNet_results, exist_ok=True)

if nnUNet_raw is None:
    print(
        "nnUNet_raw is not defined. Set the environment variable before planning, "
        "preprocessing, training, or inference."
    )

if nnUNet_preprocessed is None:
    print(
        "nnUNet_preprocessed is not defined. Set the environment variable before "
        "preprocessing or training."
    )

if nnUNet_results is None:
    print(
        "nnUNet_results is not defined. Set the environment variable before "
        "training or inference."
    )
