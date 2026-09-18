# ECG-LLaVA Phase 1 — 12-lead ECG Encoder Baseline

## One-line

PTB-XL v1.0.3의 12-lead ECG 21,388건을 누수 없는 공식 split으로 구성하고, ResNet1D encoder에서 test macro AUROC 0.9143과 32개 temporal token 인터페이스를 확인했다.

## Problem

WFDB 원본은 바로 학습에 쓰기보다 라벨 매핑, lead 순서, 길이, NaN과 split을 먼저 고정해야 했다. 특히 normalization 통계를 전체 데이터에서 계산하면 validation/test 정보가 학습 경로로 새어 들어간다. 또한 멀티라벨 분류 성능뿐 아니라 다음 단계가 사용할 encoder feature contract도 명시적으로 남길 필요가 있었다. Phase 1의 목표는 이 데이터·모델 계약을 재현 가능한 artifact로 고정하는 것이었다.

## Decision

- PTB-XL의 환자 단위 공식 `strat_fold`를 그대로 사용했다: train 1–8 / validation 9 / test 10.
- lead별 normalization 통계는 train fold에서만 계산하고, 모든 split에 같은 상수를 적용했다.
- 입력 무결성 카운트와 원본 메타데이터 SHA-256을 manifest에 기록했다.
- ResNet1D의 pooling 전 `(B, 512, 32)` feature를 32개 temporal token의 Phase 1 경계로 정의했다.

## What I Built

WFDB 신호를 `(N, 12, 1000)` float32 캐시로 변환하고, SCP 코드를 5개 superdiagnostic 멀티라벨로 매핑하는 전처리 파이프라인을 구현했다. Dataset은 memory-mapped cache와 train-only scaler를 사용한다. ResNet1D encoder, 멀티라벨 분류 head, 학습/평가 루프, 실험 config와 metrics/curve/manifest 기록 경로를 구성했다.

## Evidence

- 사용 레코드: **21,388건** (전체 21,799건 중 빈 superdiagnostic 라벨 411건 제외)
- 공식 split: **17,084 / 2,146 / 2,158** (train / validation / test)
- test macro AUROC: **0.9143**
- test macro F1: **0.7194** (`threshold=0.5`)

상세 Source of Truth는 [EVIDENCE.md](EVIDENCE.md)에 claim별로 정리했다.

## Failure / Debugging

- 원본 21,799건 중 411건은 5개 superdiagnostic 클래스에 매핑되는 라벨이 없었다. 원인은 목표 label set과 매핑되지 않는 레코드였고, 이를 무라벨 학습 샘플로 간주하지 않고 제외 수를 manifest에 남겼다.
- validation macro AUROC는 epoch 5에서 0.920266으로 최고였지만 epoch 15에는 0.908982였다. 같은 기간 train loss는 0.262617에서 0.135809로 계속 감소해 일반화 개선과 학습 loss 감소가 분리되었고, validation AUROC 기준 best checkpoint와 patience 10 early stopping으로 모델 선택을 고정했다.

## Result

저장된 test artifact 기준 5-class macro AUROC 0.9143, macro F1 0.7194를 기록했다. 전처리 manifest에는 lead 순서 불일치, NaN 포함 레코드, 길이 불일치가 각각 0건으로 기록되었다. pooling 전 feature shape `(1, 512, 32)`도 실험 artifact에 함께 저장되었다.

## Scope / Limitation

이 결과는 PTB-XL v1.0.3의 공식 test fold에 대한 단일 Phase 1 baseline이다. 외부 데이터셋 검증, 통계적 반복 실험, calibration, INT8 양자화, 배포 지연 측정은 수행하지 않았다. **Projector와 LLM은 Phase 2 범위이며 아직 구현하지 않았다.** 임상 사용 가능성이나 실제 배포 성능을 주장하지 않는다.

## Portfolio Assets

- `assets/01_ecg_pipeline.svg` / `.png`: 원본 검증부터 공식 split, train-only normalization, 분류 및 token contract까지의 데이터·모델 흐름
- `assets/02_class_auroc.svg` / `.png`: test fold의 클래스별 AUROC와 macro AUROC 기준선, macro F1 보조 지표
- `assets/03_token_interface.svg` / `.png`: `(B,12,1000)` 입력에서 `(B,512,32)` 및 32 temporal tokens까지의 Phase 1 인터페이스와 미구현 Phase 2 경계
