"""Method 2: U-Net (CNN) segmentation + ABCD features + LR / XGBoost.

1. ``python -m melanoma.method2.train_seg`` — train U-Net on Part1 images + GT masks.
2. ``python -m melanoma.method2.eval_seg`` — val Dice + IoU for a saved U-Net checkpoint.
3. ``python -m melanoma.method2.train_tabular`` — ABCD features on train split, fit LR + XGBoost.
4. ``python -m melanoma.method2.eval_test`` — test set metrics (U-Net + chosen classifier).

Use ``--use-gt-mask`` on ``train_tabular`` for an oracle upper bound (GT masks instead of U-Net).
"""
