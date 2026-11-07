# Phase 0 — Ceramic-Defect Dataset Audit Summary

Generated (UTC): 2026-11-07T08:35:11.830545+00:00

This report was produced by counting files and parsing annotations in the local archive. It does not modify original dataset files.

## Dataset location

- Data root: `E:\MTech[CSE] Nirma_University\S3\Major Project\Defect Dataset Augmentation using Generative Adversarial Networks (GAN)\Dateset\Ceramic-defect\data\data`
- Original images: `E:\MTech[CSE] Nirma_University\S3\Major Project\Defect Dataset Augmentation using Generative Adversarial Networks (GAN)\Dateset\Ceramic-defect\data\data\original image`
- Split images: `E:\MTech[CSE] Nirma_University\S3\Major Project\Defect Dataset Augmentation using Generative Adversarial Networks (GAN)\Dateset\Ceramic-defect\data\data\image`
- Labels (spelling as on disk: `lables`): JSON/XML/TXT under that folder

## Counting method

Image counts: unique filename stems under original image/ and image/{train,val,test}, extensions in IMAGE_EXTENSIONS. Annotation file counts: recursive files with .json/.xml/.txt under lables/. Instance counts: objects/shapes/lines parsed from each file. JSON LabelMe labels are the semantic class names used for per-class instance and image-containing-class counts. Image counts and instance counts are computed separately. Vendor train/val/test folders were recorded but not modified.

## File counts

- Original-folder images: **2892**
- Split-folder images: **2892**
- Unique stems: **2892**
- JSON files: **2892**
- XML files: **2892**
- TXT files: **2892**
- Images with at least one JSON object: **2892**
- Images with no JSON objects: **0**
- Vendor split (recorded, not altered): `{'train': 2023, 'val': 577, 'test': 292}`

## Annotation formats discovered

- LabelMe-style JSON rectangles (`shapes` with `shape_type=rectangle`).
- Pascal VOC XML (`object/bndbox`).
- YOLO TXT (`class_id cx cy w h`, normalized).
- All three formats share the same filename stems as the images.

## Observed class mapping

Derived by pairing objects in file order when JSON/XML/TXT instance counts matched. This is observed in this archive, not an official codebook file.

- XML name → JSON label: `{'abnormal thickness': 'abnormal thickness', 'class_0': 'adhesion', 'class_1': 'scar', 'class_2': 'spot', 'defect': 'defect', 'uneven': 'uneven'}`
- YOLO id → JSON label: `{'0': 'adhesion', '1': 'scar', '2': 'spot', '3': 'uneven', '4': 'abnormal thickness', '5': 'defect'}`

## Per-class counts (JSON semantic labels)

| Class | Images containing class | Bounding-box instances |
|---|---:|---:|
| spot | 908 | 3165 |
| scar | 763 | 1593 |
| adhesion | 632 | 1270 |
| defect | 808 | 1137 |
| uneven | 368 | 368 |
| abnormal thickness | 156 | 156 |

- JSON instance total: **7689**
- Multi-class images: **637**
- Instance imbalance (max/min): **20.28846153846154**

## Image statistics

- Formats: `{'JPEG': 2892}`
- Width: `{'count': 2892, 'min': 1024.0, 'max': 4096.0, 'mean': 1428.478561549101, 'median': 1024.0, 'p25': 1024.0, 'p75': 1024.0}`
- Height: `{'count': 2892, 'min': 1024.0, 'max': 4300.0, 'mean': 1427.899723374827, 'median': 1024.0, 'p25': 1024.0, 'p75': 1024.0}`
- Aspect ratio (W/H): `{'count': 2892, 'min': 0.8717509283061983, 'max': 1.024, 'mean': 1.0001968323686423, 'median': 1.0, 'p25': 1.0, 'p75': 1.0}`
- Resolution histogram: `{'1024x1024': 2498, '4096x4000': 245, '4096x4300': 63, '4000x4096': 62, '2107x2058': 4, '2058x2107': 4, '2026x2038': 2, '2038x2026': 2, '3052x3501': 1, '3200x3509': 1, '3194x3477': 1, '3206x3506': 1, '3200x3492': 1, '3370x3525': 1, '3115x3484': 1, '3318x3506': 1, '3276x3467': 1, '3339x3491': 1, '3289x3456': 1, '3293x3480': 1}`
- Unreadable images: **0**
- Unsupported formats: **0**
- Filename prefix buckets: `{'numeric_id': 1361, 'defect': 808, 'uneven': 368, 'Abnormal thickness': 156, 'adhension': 79, 'spot': 48, 'con': 22, 'crack': 21, 'scar': 18, 'dirty': 11}`

## Bounding-box statistics (from JSON)

- COCO-style size bins: `{'definition': {'small': 'area < 1024', 'medium': '1024 <= area < 9216', 'large': 'area >= 9216'}, 'counts': {'medium': 4280, 'small': 1987, 'large': 1422}}`
- Width px: `{'count': 7689, 'min': 6.727828746177238, 'max': 3845.454545454546, 'mean': 100.91431713213605, 'median': 44.44444444444446, 'p25': 28.888888888888857, 'p75': 78.94736842105272}`
- Height px: `{'count': 7689, 'min': 6.727828746177352, 'max': 3795.6521739130435, 'mean': 89.70090372517951, 'median': 45.55555555555554, 'p25': 30.0, 'p75': 75.36231884057997}`
- Area px: `{'count': 7689, 'min': 45.26367963788955, 'max': 5601852.551984877, 'mean': 34015.64142843193, 'median': 1925.9259259259327, 'p25': 999.9999999999977, 'p75': 5629.629629629635}`
- Relative area: `{'count': 7689, 'min': 2.5699309387428207e-06, 'max': 0.8713522075135031, 'mean': 0.01246004799246986, 'median': 0.0014505268614969074, 'p25': 0.0006769910270785147, 'p75': 0.003223654664592979}`

## Validation issue counts

`{}`

- Annotation/image dimension mismatches: **0**
- Cross-format instance-count mismatches: **0**
- Duplicate annotation entry groups: **0**

## Duplicates and related views

Potential near-duplicates: 8x8 average-hash Hamming distance <= 5, excluding confirmed exact MD5 duplicates. These are candidates only.

- Confirmed exact MD5 duplicate groups (within original image/): **8**
- Potential near-duplicate pairs: **200**
- Original vs split copies compared: **2892**
- Original vs split hash mismatches: **0**

original image/ and image/{{train,val,test}} contain the same filename stems. Identical MD5 hashes mean two copies of the same file, not two distinct specimens.

- Filename groups with multiple members: **302**
- Vendor-split leakage candidates (same filename group in more than one vendor split): **50**

## Visualizations

- `E:\MTech[CSE] Nirma_University\S3\Major Project\Defect Dataset Augmentation using Generative Adversarial Networks (GAN)\outputs\audit\phase0\visualizations\class_abnormal_thickness_Abnormal_thickness001.jpg`
- `E:\MTech[CSE] Nirma_University\S3\Major Project\Defect Dataset Augmentation using Generative Adversarial Networks (GAN)\outputs\audit\phase0\visualizations\class_abnormal_thickness_Abnormal_thickness002.jpg`
- `E:\MTech[CSE] Nirma_University\S3\Major Project\Defect Dataset Augmentation using Generative Adversarial Networks (GAN)\outputs\audit\phase0\visualizations\class_adhesion_1_2_000001_0810104210546_NG_part0.jpg`
- `E:\MTech[CSE] Nirma_University\S3\Major Project\Defect Dataset Augmentation using Generative Adversarial Networks (GAN)\outputs\audit\phase0\visualizations\class_defect_defect001.jpg`
- `E:\MTech[CSE] Nirma_University\S3\Major Project\Defect Dataset Augmentation using Generative Adversarial Networks (GAN)\outputs\audit\phase0\visualizations\class_defect_defect002.jpg`
- `E:\MTech[CSE] Nirma_University\S3\Major Project\Defect Dataset Augmentation using Generative Adversarial Networks (GAN)\outputs\audit\phase0\visualizations\class_scar_1_1_000004_0121102105699_NG.jpg`
- `E:\MTech[CSE] Nirma_University\S3\Major Project\Defect Dataset Augmentation using Generative Adversarial Networks (GAN)\outputs\audit\phase0\visualizations\class_spot_1_1_000004_0121102105699_NG.jpg`
- `E:\MTech[CSE] Nirma_University\S3\Major Project\Defect Dataset Augmentation using Generative Adversarial Networks (GAN)\outputs\audit\phase0\visualizations\class_uneven_uneven001.jpg`
- `E:\MTech[CSE] Nirma_University\S3\Major Project\Defect Dataset Augmentation using Generative Adversarial Networks (GAN)\outputs\audit\phase0\visualizations\class_uneven_uneven002.jpg`
- `E:\MTech[CSE] Nirma_University\S3\Major Project\Defect Dataset Augmentation using Generative Adversarial Networks (GAN)\outputs\audit\phase0\visualizations\small_defect_1_1_000004_0121102105699_NG.jpg`
- `E:\MTech[CSE] Nirma_University\S3\Major Project\Defect Dataset Augmentation using Generative Adversarial Networks (GAN)\outputs\audit\phase0\visualizations\large_defect_1_1_000004_0121102105699_NG.jpg`
- `E:\MTech[CSE] Nirma_University\S3\Major Project\Defect Dataset Augmentation using Generative Adversarial Networks (GAN)\outputs\audit\phase0\visualizations\large_defect_1_1_000005_0121102216518_NG.jpg`
- `E:\MTech[CSE] Nirma_University\S3\Major Project\Defect Dataset Augmentation using Generative Adversarial Networks (GAN)\outputs\audit\phase0\visualizations\large_defect_1_2_000001_0813113715924_NG_full.jpg`
- `E:\MTech[CSE] Nirma_University\S3\Major Project\Defect Dataset Augmentation using Generative Adversarial Networks (GAN)\outputs\audit\phase0\visualizations\large_defect_1_2_000002_0121101850711_NG.jpg`

## License status

- Status: **Unverified in automated pass**
- Detail: Automated search results are in audit_report.json under license_files_found_in_dataset_tree.

## Suitability verdict

**See the Phase 0 chat report / updated Markdown after qualitative review.**

Numeric audit values are in this folder. The research suitability verdict is written after reviewing those values and license files.

## Tests

- Command: `pytest`
- Result: Run separately; see test_results.txt

## Commands executed

- `python scripts/audit_ceramic_defect.py --dataset-root E:\MTech[CSE] Nirma_University\S3\Major Project\Defect Dataset Augmentation using Generative Adversarial Networks (GAN)\Dateset\Ceramic-defect\data\data --output-dir E:\MTech[CSE] Nirma_University\S3\Major Project\Defect Dataset Augmentation using Generative Adversarial Networks (GAN)\outputs\audit\phase0`
