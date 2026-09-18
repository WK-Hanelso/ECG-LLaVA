# Evidence Register

모든 성능 수치는 기존 artifact를 전사했으며, 이 package를 만들기 위해 학습 또는 평가를 새로 실행하지 않았다. 표의 `Reproducible`은 저장된 source에서 claim과 asset을 다시 추출할 수 있다는 뜻이며, 원 데이터나 체크포인트의 외부 공개 여부를 뜻하지 않는다.

| Claim | Value | Source file/log | Reproducible | Public safe | Note |
|---|---|---|---|---|---|
| C01 — 사용 데이터 규모 | 21,388 used / 21,799 total / 411 dropped | `results/resnet1d_superdiag_base/prepare_manifest.json` → `n_used`, `n_total_records`, `n_dropped_empty_label` | Yes | Yes | 목표 superdiagnostic 라벨이 비어 있는 레코드 제외 |
| C02 — 공식 split | train 17,084 (fold 1–8), val 2,146 (fold 9), test 2,158 (fold 10) | `results/resnet1d_superdiag_base/prepare_manifest.json` → `split` | Yes | Yes | PTB-XL `strat_fold` 사용 |
| C03 — 입력 cache contract | `(21388, 12, 1000)`, float32, 100 Hz | `results/resnet1d_superdiag_base/prepare_manifest.json` → `signal_shape`, `signal_dtype`, `sampling_rate` | Yes | Yes | 10초 12-lead 입력 |
| C04 — 입력 무결성 결과 | lead-order mismatch 0 / NaN records 0 / length mismatch 0 | `results/resnet1d_superdiag_base/prepare_manifest.json` → `n_lead_order_mismatch`, `n_records_with_nan`, `n_length_mismatch` | Yes | Yes | 기존 전처리 실행의 manifest 결과 |
| C05 — train-only normalization | scaler statistics computed on train folds 1–8 only | `src/ecgllava/data/prepare_ptbxl.py:125`; `src/ecgllava/data/prepare_ptbxl.py:153`; `src/ecgllava/data/dataset.py:38` | Yes | Yes | leakage 방지를 위한 코드 경로 |
| C06 — test macro AUROC | 0.9143272980 (display: 0.9143) | `results/resnet1d_superdiag_base/metrics.json` → `test.macro_auroc` | Yes | Yes | 5개 클래스 단순 평균 |
| C07 — test macro F1 | 0.7194360544 (display: 0.7194), threshold 0.5 | `results/resnet1d_superdiag_base/metrics.json` → `test.macro_f1`, `test.threshold` | Yes | Yes | 고정 threshold의 보조 지표 |
| C08 — class AUROC | NORM 0.9347 / MI 0.9022 / STTC 0.9311 / CD 0.9080 / HYP 0.8956 | `results/resnet1d_superdiag_base/metrics.json` → `test.per_class_auroc` | Yes | Yes | 화면 표기는 소수점 넷째 자리 반올림 |
| C09 — token interface | `(B, 512, 32)` = 32 temporal tokens | `src/ecgllava/models/resnet1d.py:99`; `results/resnet1d_superdiag_base/metrics.json` → `feature_shape` | Yes | Yes | artifact의 batch example은 `(1,512,32)` |
| C10 — 모델 규모 | 8,739,973 parameters | `results/resnet1d_superdiag_base/metrics.json` → `n_params` | Yes | Yes | ResNet1D + classification head |
| C11 — model selection | best epoch 5 / 15 epochs run | `results/resnet1d_superdiag_base/metrics.json` → `best_epoch`, `epochs_run` | Yes | Yes | validation macro AUROC 기준 |
| C12 — epoch 5 이후 validation 저하 | val AUROC 0.920266 → 0.908982; train loss 0.262617 → 0.135809 (epoch 5 → 15) | `results/resnet1d_superdiag_base/curves.csv:6`; `results/resnet1d_superdiag_base/curves.csv:16` | Yes | Yes | early stopping과 best checkpoint의 근거 |
| C13 — manifest provenance | source metadata SHA-256 values recorded | `results/resnet1d_superdiag_base/prepare_manifest.json` → `sha256` | Yes | Yes | 공개 파일명과 digest만 사용; 원본 절대 경로는 미노출 |
| C14 — Phase 2 boundary | Projector + LLM not built | `README.md:97`; `README.md:101`; `src/ecgllava/models/resnet1d.py:99` | Yes | Yes | Phase 1은 encoder feature contract까지만 포함 |

## Artifact integrity

- `metrics.json` SHA-256: `57a0d30be7cf897ceb79ab51f6c963891208b4e2b5bb54e8d9f860fdeba015a4`
- `prepare_manifest.json` SHA-256: `e05339ff63e651544683c2545d668aa02952944d1221a8edaf9357cdeb16bff4`
- `curves.csv` SHA-256: `250eb0099f91d1f1002347f79548e2385769cd36ce1da64b3fba3ba8cafee8de`
