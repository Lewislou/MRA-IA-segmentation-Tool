import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import label


def load_nii(path):
    img = nib.load(path)
    return img.get_fdata(), img.affine, img.header


def connected_components(mask):
    labeled, n = label(mask.astype(np.uint8))
    comps = []
    for i in range(1, n + 1):
        coords = np.argwhere(labeled == i)
        if coords.size == 0:
            continue
        comps.append({"id": i, "coords": coords, "size": int(coords.shape[0])})
    return comps


def iou_3d(a_coords, b_coords):
    a = set(map(tuple, a_coords))
    b = set(map(tuple, b_coords))
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / float(len(a | b))


def match_case(gt_mask, pred_mask, iou_thr, min_pred_size):
    gt_comps = connected_components(gt_mask > 0)
    pred_comps = [c for c in connected_components(pred_mask > 0) if c["size"] >= min_pred_size]
    matched_gt = set()
    tp = 0
    for pc in pred_comps:
        best_iou, best_g = 0.0, None
        for gc in gt_comps:
            if gc["id"] in matched_gt:
                continue
            v = iou_3d(pc["coords"], gc["coords"])
            if v > best_iou:
                best_iou, best_g = v, gc["id"]
        if best_iou >= iou_thr and best_g is not None:
            tp += 1
            matched_gt.add(best_g)
    fp = len(pred_comps) - tp
    fn = len(gt_comps) - len(matched_gt)
    return tp, fp, fn


def process_one(args):
    case_id, gt_path, pred_path, iou_thr, min_pred_size = args
    gt, _, _ = load_nii(gt_path)
    pred, _, _ = load_nii(pred_path)
    if pred.shape != gt.shape:
        raise ValueError(f"{case_id}: shape mismatch {pred.shape} vs {gt.shape}")
    tp, fp, fn = match_case(gt, pred, iou_thr, min_pred_size)
    return {
        "case_id": case_id,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "sensitivity": tp / (tp + fn) if (tp + fn) else np.nan,
        "precision": tp / (tp + fp) if (tp + fp) else np.nan,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_folder", required=True)
    parser.add_argument("--pred_folder", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    parser.add_argument("--min_pred_size", type=int, default=50)
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() // 2))
    args = parser.parse_args()

    gt_files = {f.replace(".nii.gz", ""): os.path.join(args.gt_folder, f)
                for f in os.listdir(args.gt_folder) if f.endswith(".nii.gz")}
    pred_files = {f.replace(".nii.gz", ""): os.path.join(args.pred_folder, f)
                  for f in os.listdir(args.pred_folder) if f.endswith(".nii.gz")}
    common = sorted(set(gt_files) & set(pred_files))
    if not common:
        raise RuntimeError("No overlapping case ids between gt_folder and pred_folder")

    jobs = [
        (cid, gt_files[cid], pred_files[cid], args.iou_threshold, args.min_pred_size)
        for cid in common
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        rows = list(ex.map(process_one, jobs))

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)) or ".", exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    tp, fp, fn = df["tp"].sum(), df["fp"].sum(), df["fn"].sum()
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    f1 = 2 * sens * prec / (sens + prec) if (sens + prec) else float("nan")
    print(f"cases={len(df)} tp={tp} fp={fp} fn={fn} sens={sens:.4f} prec={prec:.4f} f1={f1:.4f}")
    print(f"wrote {args.output_csv}")


if __name__ == "__main__":
    main()
