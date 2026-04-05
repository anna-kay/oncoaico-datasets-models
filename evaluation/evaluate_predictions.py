import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

# =========================
# Dataset Loader
# =========================
class TestDataset:
    def __init__(self, pred_root, gt_root):
        pred_mask_root = os.path.join(pred_root, "masks")
        gt_mask_root = os.path.join(gt_root, "masks")

        self.pred_paths = sorted([
            os.path.join(pred_mask_root, f)
            for f in os.listdir(pred_mask_root)
            if f.endswith(('.png', '.jpg'))
        ])

        self.gt_paths = sorted([
            os.path.join(gt_mask_root, f)
            for f in os.listdir(gt_mask_root)
            if f.endswith(('.png', '.jpg'))
        ])

        assert len(self.pred_paths) == len(self.gt_paths), \
            f"Mismatch: {len(self.pred_paths)} preds vs {len(self.gt_paths)} gts"

    def __len__(self):
        return len(self.pred_paths)

    def __getitem__(self, idx):
        pred = np.array(Image.open(self.pred_paths[idx]).convert('L'), dtype=np.float32)
        gt = np.array(Image.open(self.gt_paths[idx]).convert('L'), dtype=np.float32)

        pred = pred / 255.0
        gt = (gt > 127).astype(np.float32)

        name = os.path.basename(self.pred_paths[idx])
        return pred, gt, name


# =========================
# Evaluator
# =========================
class Evaluator:
    def __init__(self, threshold=0.5, eps=1e-8, min_area=0):
        self.th = threshold
        self.eps = eps
        self.min_area = min_area
        self.reset()

    def reset(self):
        self.dice_sum = 0
        self.iou_sum = 0
        self.prec_sum = 0
        self.rec_sum = 0
        self.pos_count = 0

        self.tp = self.tn = self.fp = self.fn = 0

        self.neg_total = 0
        self.tn_only = 0
        self.fp_only = 0

    def _binarize(self, pred):
        return (pred > self.th).astype(np.float32)

    def _is_positive(self, mask):
        return mask.sum() > self.min_area

    def update(self, pred, gt):
        pred_bin = self._binarize(pred)

        pred_flat = pred_bin.flatten()
        gt_flat = gt.flatten()

        TP_px = np.sum((pred_flat == 1) & (gt_flat == 1))
        FP_px = np.sum((pred_flat == 1) & (gt_flat == 0))
        FN_px = np.sum((pred_flat == 0) & (gt_flat == 1))

        gt_positive = self._is_positive(gt_flat)
        pred_positive = self._is_positive(pred_flat)

        if gt_positive:
            self.pos_count += 1

            self.dice_sum += (2 * TP_px) / (2 * TP_px + FP_px + FN_px + self.eps)
            self.iou_sum += TP_px / (TP_px + FP_px + FN_px + self.eps)
            self.prec_sum += TP_px / (TP_px + FP_px + self.eps)
            self.rec_sum += TP_px / (TP_px + FN_px + self.eps)

        if gt_positive and pred_positive:
            self.tp += 1
        elif not gt_positive and not pred_positive:
            self.tn += 1
        elif not gt_positive and pred_positive:
            self.fp += 1
        elif gt_positive and not pred_positive:
            self.fn += 1

        if not gt_positive:
            self.neg_total += 1
            if not pred_positive:
                self.tn_only += 1
            else:
                self.fp_only += 1

    def compute(self):
        mDice = self.dice_sum / self.pos_count if self.pos_count > 0 else 0
        mIoU = self.iou_sum / self.pos_count if self.pos_count > 0 else 0
        Prec_px = self.prec_sum / self.pos_count if self.pos_count > 0 else 0
        Recall_px = self.rec_sum / self.pos_count if self.pos_count > 0 else 0

        total = self.tp + self.tn + self.fp + self.fn

        Sens = self.tp / (self.tp + self.fn + self.eps)
        Spec = self.tn / (self.tn + self.fp + self.eps)
        Acc = (self.tp + self.tn) / (total + self.eps)
        F1 = (2 * self.tp) / (2 * self.tp + self.fp + self.fn + self.eps)

        FPR = self.fp_only / (self.neg_total + self.eps)

        total_pos = self.tp + self.fn
        FNR = self.fn / total_pos if total_pos > 0 else 0.0

        return {
            "mDice": mDice,
            "mIoU": mIoU,
            "Prec_px": Prec_px,
            "Recall_px": Recall_px,
            "Sens": Sens,
            "Spec": Spec,
            "Acc": Acc,
            "F1": F1,
            "FPR": FPR,
            "FNR": FNR
        }


# =========================
# Evaluation
# =========================
def evaluate(pred_root, gt_root, threshold, min_area):
    dataset = TestDataset(pred_root, gt_root)
    evaluator = Evaluator(threshold=threshold, min_area=min_area)

    for i in tqdm(range(len(dataset)), desc=f"Evaluating {os.path.basename(pred_root)}"):
        pred, gt, _ = dataset[i]
        evaluator.update(pred, gt)

    return evaluator.compute()


# =========================
# Pretty Print
# =========================
def print_results(name, results):
    print("\n" + "=" * 60)
    print(f"Dataset: {name}")
    print("=" * 60)

    print("\n[Segmentation - Positives Only]")
    print(f"mDice:   {results['mDice']:.4f}")
    print(f"mIoU:    {results['mIoU']:.4f}")
    print(f"Recall:  {results['Recall_px']:.4f}")
    print(f"Prec:    {results['Prec_px']:.4f}")

    print("\n[Detection - All Images]")
    print(f"Sens: {results['Sens']:.4f}")
    print(f"Spec: {results['Spec']:.4f}")
    print(f"Acc:  {results['Acc']:.4f}")
    print(f"F1:   {results['F1']:.4f}")

    print("\n[Error Rates]")
    print(f"FPR:  {results['FPR']:.4f}")
    print(f"FNR:  {results['FNR']:.4f}")

    print("=" * 60)


# =========================
# CLI
# =========================
def main():
    parser = argparse.ArgumentParser(description="Batch Evaluation Script")

    parser.add_argument("--pred_base", type=str, required=True,
                        help="Base path containing dataset subfolders")
    parser.add_argument("--gt_base", type=str, required=True,
                        help="Base GT path containing dataset subfolders")

    parser.add_argument("--datasets", nargs="+", required=True,
                        help="List of dataset names (e.g. negatives positives mixed_testsets)")

    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min_area", type=int, default=0)

    args = parser.parse_args()

    all_results = {}

    for d in args.datasets:
        pred_root = os.path.join(args.pred_base, d)
        gt_root = os.path.join(args.gt_base, d)

        results = evaluate(pred_root, gt_root, args.threshold, args.min_area)
        print_results(d, results)

        all_results[d] = results

    return all_results


if __name__ == "__main__":
    main()