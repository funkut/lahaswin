# lahaswin — LAHA-Swin: Multi-Scale Dilated Attention with Adaptive Gating for Maize Foliar Disease Classification

Code, exact fold indices, and every out-of-fold prediction file behind the numbers
reported in the manuscript (Applied Soft Computing, ASOC-D-26-08825, revised version).

## What is here

```
config.py                 hyper-parameters, seeding (set LAHA_DATA / LAHA_OUT)
data/dataset.py           loader (jpg/png/tif/webp), transforms
models/laha_swin.py       LAHA module, SE/ECA/CBAM zoo, SwinWithAttention, build_model
models/laha_variants.py   structural ablation variants (Table 13)
train.py                  train_one_fold (primary 40E) and _quick_train (20E SOTA/ablation)
stats/compute_stats.py    reproduces every paired test from results/  -> paired_tests.{json,md}
splits/                   exact fold indices (group-aware) used for all reported numbers
results/                  out-of-fold predictions (preds, labels, probs, test_idx) per fold
experiments/              reference notebooks
```

## Two evaluation protocols (do not mix them)

| Protocol | Function | Epochs | Inner val split | Used for |
|---|---|---|---|---|
| **Primary** | `train_one_fold` | 40 + early stopping | 0.15 | headline result, Tables 12, 13, 15, calibration, Grad-CAM |
| **Quick-train (SOTA / tricks / PV ablation)** | `_quick_train` | 20 | 0.20 | Tables 10, 14, 23, 25 |

Absolute values from the two protocols are not directly comparable; each table
caption states which one it uses.

## Fold sets

All splits are 10-fold and **group-aware**: near-duplicate groups never straddle a
fold boundary. Two group-aware partitions exist on Maize:

- **Option B — broad, minimal repair** (`splits/folds_maize.json`, fold test sizes
  1224 / 1201 / 1556 / …): all 994 hash-flagged candidate pairs grouped, 414 images
  relocated from the original stratified split. **This is the fold set of every
  LAHA-Swin result and of all paired tests.** `master_compatible_split_manifest.json`
  gives the same folds with the train/val/test lists of the primary run.
- **Option A — StratifiedGroupKFold** (equal fold sizes, ~1231): used for the
  quick-train SOTA baselines in `results/baselines_optA/`. These cannot be paired
  with LAHA-Swin at instance level because the test samples differ.

`group_aware_maize.json` / `group_aware_pv.json` contain the candidate pairs,
group ids, both options and the relocation counts.

## Table → result files

| Table | Content | Folder | Protocol / folds |
|---|---|---|---|
| headline, T8/T9 | LAHA-Swin learnable α | `results/main_LAHA/` | primary, Opt B |
| **T12** attention ablation | SE, ECA, CBAM | `results/attention_ablation/` | primary, Opt B |
| **T13** component ablation | channel/spatial/no_d2/no_d3, no-gate, fixed α | `results/component_ablation/`, `nogate/`, `fixed_alpha/` | primary, Opt B |
| **T15** paired tests | Swin-Tiny, Swin-Small retrained on Opt B | `results/baselines_optB/` | primary, Opt B |
| T16 multi-seed | seeds 1337, 2024 | `results/multiseed/` | primary, Opt B |
| T10 / T23 SOTA | ConvNeXt, ViT, EfficientNet (fairness-tuned) | `results/baselines_optA/` | quick-train, Opt A |
| Grad-CAM (200 img) | deletion/insertion + random-map control | `results/gradcam/` | — |

Every `*_foldN.json` has `preds`, `labels`, `probs` (where available) and
`test_idx` (global sample index), so any pooled out-of-fold quantity — accuracy,
macro-F1, ECE, AURC, selective accuracy — can be recomputed.

## Reproducing the statistics

```bash
pip install -r requirements.txt
python stats/compute_stats.py
```

writes `stats/paired_tests.json` and `stats/paired_tests.md`: raw and
Holm-adjusted McNemar and Wilcoxon p-values, rank-biserial r, discordant-pair
counts and fold-level win/loss/tie for every comparison in Tables 12, 13 and 15,
with Holm applied within each table family (3, 10 and 2 comparisons respectively).

Sign convention in the script: `delta` and `r` are variant − reference. Table 15 in
the manuscript reports r with the opposite sign (positive = LAHA-Swin higher); the
magnitudes are identical.

## Training

```bash
export LAHA_DATA=/path/to/data_maize_diseases/archive
python - <<'PY'
from config import *; from data.dataset import *; from models.laha_swin import *; from train import *
import json, numpy as np
samples, labels = load_dataset(CFG.V11_DIR)
folds = json.load(open('splits/folds_maize.json'))
# ... build train/val/test from folds[i]['test_idx'] with a 0.15 stratified inner split,
#     then train_one_fold(i, train, val, test, attn_type='laha')
PY
```

For the structural variants: `from models.laha_variants import register; register(ATTENTION_ZOO)`
then `attn_type` in `{'channel_only','spatial_single','spatial_multi','no_d2','no_d3'}`.

## Inner-split note

Outer test folds are group-aware. The inner train/validation split inside each fold
is class-stratified but not group-aware (identical across all configurations for
comparability). Because every reported metric is computed on the group-aware outer
test folds, near-duplicate leakage cannot affect the reported scores; the inner split
only influences early-stopping epoch selection.

## Checkpoints

Not stored in git. The ten primary LAHA-Swin fold checkpoints (~1.9 GB) are on
Zenodo (DOI to be added).

## Citation / licence

MIT. Please cite the manuscript.
