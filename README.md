# lahaswin

LAHA-Swin: a lesion-aware hybrid attention module (gated channel calibration +
multi-scale dilated spatial attention + learnable residual scaling) on a Swin-Small
backbone, for maize foliar disease classification.

This repository holds the code, the exact group-aware fold indices, and every
out-of-fold prediction file behind the reported results, so that all numbers and
paired statistics can be recomputed without retraining.

## Layout

```
config.py                 hyper-parameters, seeding
data/dataset.py           loader and transforms
models/laha_swin.py       LAHA module, SE/ECA/CBAM baselines, build_model
models/laha_variants.py   structural ablation variants
train.py                  40-epoch primary protocol and 20-epoch quick-train protocol
stats/compute_stats.py    paired McNemar / Wilcoxon / Holm, effect sizes, win/loss/tie
splits/                   group-aware 10-fold indices (Maize, PlantVillage)
results/                  per-fold predictions (preds, labels, probs, test_idx)
experiments/              notebooks used for each run
```

## Reproduce the statistics

```bash
pip install -r requirements.txt
python stats/compute_stats.py      # -> stats/paired_tests.{json,md}
```

## Data

Not redistributed. Place Maize Diseases v1.1 and the PlantVillage corn subset under
`data_maize_diseases/archive/` and set `LAHA_DATA` (see `data/README_data.md`).
`splits/` fixes the fold assignment; near-duplicate groups never cross a fold boundary.

## Checkpoints

Trained weights are hosted on Zenodo (link to be added).

## Licence

MIT.
