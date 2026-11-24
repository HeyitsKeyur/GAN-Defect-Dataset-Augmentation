# SolDef_AI Primary Side-View Benchmark Split Summary

- **Benchmark:** Solder Joint Defect Detection (Side Views: V2, V2.1)
- **Total Images:** 200 (Train: 140 [70.0%], Val: 30 [15.0%], Test: 30 [15.0%])
- **Total Independent Groups:** 66 (Train: 30, Val: 19, Test: 17)
- **Seed:** 0
- **Cross-Split Leakage:** 0 image overlap, 0 group overlap (VERIFIED LEAKAGE-FREE)

## Class Distribution Across Splits

| Class | Train Instances (Images) | Val Instances (Images) | Test Instances (Images) | Total Instances |
| :--- | :--- | :--- | :--- | :--- |
| `exc_solder` | 128 (101) | 26 (21) | 28 (19) | 182 |
| `spike` | 96 (81) | 15 (14) | 19 (19) | 130 |
| `poor_solder` (minority) | 32 (27) | 10 (10) | 15 (14) | 57 |
| `good` (normal baseline) | 50 (43) | 14 (14) | 10 (10) | 74 |
| **Total Annotations** | **306** | **65** | **72** | **443** |
