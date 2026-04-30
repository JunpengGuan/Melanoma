import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from melanoma.method1.data_mask_concat import LesionImageMaskConcatDataset
from melanoma.method1.data_mask_concat import filter_existing_rows, load_rows
from melanoma.method1.models_mask_concat import build_classifier_mask_concat


@torch.no_grad()
def run_eval(model, loader, device):
    model.eval()

    probs = []
    labels = []

    for x, y in loader:
        x = x.to(device)

        logits = model(x)
        prob = torch.sigmoid(logits).squeeze(-1).cpu()

        probs += prob.tolist()
        labels += y.int().tolist()

    return probs, labels


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--label-csv", type=Path, required=True)
    parser.add_argument("--backbone", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.5)

    args = parser.parse_args()

    from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    if args.backbone is not None:
        backbone = args.backbone
    else:
        backbone = ckpt["backbone"]

    all_rows = load_rows(args.label_csv)
    rows = filter_existing_rows(all_rows, args.image_dir, args.mask_dir)

    if len(rows) == 0:
        raise ValueError("No image/mask pairs found. Check paths.")

    skipped = len(all_rows) - len(rows)

    dataset = LesionImageMaskConcatDataset(
        rows,
        args.image_dir,
        args.mask_dir,
        train=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = build_classifier_mask_concat(backbone, pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.to(device)

    probs, y_true = run_eval(model, loader, device)

    y_pred = []
    for p in probs:
        if p >= args.threshold:
            y_pred.append(1)
        else:
            y_pred.append(0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

    if len(set(y_true)) >= 2:
        auc = float(roc_auc_score(y_true, probs))
    else:
        auc = float("nan")

    print("samples:", len(y_true), " device:", device, " backbone:", backbone)
    print("skipped missing image or mask:", skipped)
    print("accuracy @", args.threshold, ":", round(float(accuracy_score(y_true, y_pred)), 4))
    print("sensitivity (melanoma recall):", round(sensitivity, 4))
    print("specificity:", round(specificity, 4))
    print("confusion_matrix [tn fp; fn tp]:")
    print("  TN=", int(tn), "FP=", int(fp))
    print("  FN=", int(fn), "TP=", int(tp))
    print("auc_roc:", round(auc, 4))


if __name__ == "__main__":
    main()