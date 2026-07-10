# Data Card — MVTec AD

## Source
MVTec Anomaly Detection dataset, MVTec Software GmbH.
https://www.mvtec.com/company/research/datasets/mvtec-ad
License: **CC BY-NC-SA 4.0** — non-commercial research use only. Cite Bergmann
et al., "MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly
Detection," CVPR 2019.

## What it is
- ~5,350 high-resolution images across 15 categories (objects + textures):
  bottle, cable, capsule, carpet, grid, hazelnut, leather, metal_nut, pill,
  screw, tile, toothbrush, transistor, wood, zipper.
- **Train**: defect-free ("good") images only — exactly the one-class setting.
- **Test**: good images plus images with real defects (scratches, dents,
  contamination, misalignment, …), with per-defect subfolders.
- The standard real-world benchmark for unsupervised anomaly detection.

## Download (manual — license click-through, not auto-downloadable)
1. Visit the source page and accept the license.
2. Download and extract so the layout is:
```
data/mvtec_anomaly_detection/
  <category>/
    train/good/*.png
    test/good/*.png
    test/<defect_type>/*.png
    ground_truth/<defect_type>/*_mask.png   # pixel masks (optional, for segmentation)
```
3. Use `get_mvtec("<category>")` from `src/data.py`.

## How we use it
- Train on `train/good` only (label 0).
- Test: `good` → label 0, any defect subfolder → label 1.
- Images converted to grayscale and resized (default 256×256) for the shared
  models; increase resolution if compute allows.

## Notes / limitations
- Few-shot regime: some categories have only a few dozen training images.
- Defects are subtle and localized — image-level AUROC is harder than on
  Fashion-MNIST, which is the point of including it.
- Ground-truth masks enable pixel-level evaluation; out of scope for our
  image-level comparison but a natural extension.
- Non-commercial license: do not redistribute the data with the repo; the
  reproducibility guide points graders to the official download instead.
