# Phase 1 — Group-Aware Split Summary

**Seed:** 42  
**Ratios (group-level):** train=70% · val=15% · test=15%  
**Total images:** 2892  
**Total groups:** 1997  
**Vendor-split leakage groups corrected:** 189  

## Image counts per split

| Split | Images |
|---|---:|
| train | 1967 |
| val | 479 |
| test | 446 |

## Instance counts per class per split

| Class | train | val | test |
|---|---:|---:|---:|
| abnormal_thickness | 108 | 25 | 23 |
| adhesion | 852 | 202 | 216 |
| defect | 820 | 156 | 161 |
| scar | 1133 | 257 | 203 |
| spot | 2088 | 589 | 488 |
| uneven | 261 | 52 | 55 |

## Image counts per class per split

| Class | train | val | test |
|---|---:|---:|---:|
| abnormal_thickness | 108 | 25 | 23 |
| adhesion | 425 | 107 | 100 |
| defect | 573 | 117 | 118 |
| scar | 504 | 145 | 114 |
| spot | 591 | 169 | 148 |
| uneven | 261 | 52 | 55 |

## Group size histogram

| Members per group | Number of groups |
|---:|---:|
| 1 | 1690 |
| 2 | 125 |
| 3 | 48 |
| 4 | 47 |
| 5 | 27 |
| 6 | 17 |
| 7 | 11 |
| 8 | 15 |
| 9 | 4 |
| 10 | 5 |
| 11 | 2 |
| 12 | 3 |
| 13 | 1 |
| 14 | 1 |
| 15 | 1 |

## Warnings

- Vendor split had 189 groups crossing train/val/test boundaries. These are corrected in this group-aware split.
- 8 confirmed MD5 duplicate pairs merged into shared groups.

## Safety notes

- The **test split is real-only**. Never use test images for GAN training, synthetic sample selection, or hyperparameter tuning.
- The **GAN must be trained only on the train split**.
- Split manifests are frozen. Do not regenerate without documenting the reason and changing the seed or output directory.
- Original dataset files were not modified.
