# Portfolio Asset Captions

## 01_ecg_pipeline.svg / 01_ecg_pipeline.png

Notion 위치: `Decision` 다음, `What I Built` 시작 부분.

Caption: PTB-XL WFDB 원본을 5-class 멀티라벨과 고정 shape cache로 변환하고, 공식 `strat_fold` 및 train-only normalization을 적용한 Phase 1 파이프라인. 기존 manifest에는 21,388건의 `(12,1000)` 입력과 lead 순서·NaN·길이 불일치 0건이 기록되어 있다.

Claim boundary: 도식의 숫자는 기존 `prepare_manifest.json`과 `metrics.json`에 기록된 값만 사용했다. 실시간 처리, 배포 성능, 외부 데이터 일반화는 나타내지 않는다. Projector/LLM 구현을 포함하지 않는다.

단계 ↔ 코드/근거:

| Stage | Module / artifact |
|---|---|
| PTB-XL WFDB | `src/ecgllava/data/prepare_ptbxl.py:49` |
| Integrity check | `src/ecgllava/data/prepare_ptbxl.py:98`; `src/ecgllava/data/prepare_ptbxl.py:108`; `results/resnet1d_superdiag_base/prepare_manifest.json` |
| Label mapping | `src/ecgllava/data/labels.py:18`; `src/ecgllava/data/labels.py:34` |
| Fixed-shape cache | `src/ecgllava/data/prepare_ptbxl.py:98`; `src/ecgllava/data/prepare_ptbxl.py:134` |
| Official `strat_fold` | `src/ecgllava/data/prepare_ptbxl.py:175`; `configs/base.py:20` |
| Train-only normalization | `src/ecgllava/data/prepare_ptbxl.py:125`; `src/ecgllava/data/dataset.py:38` |
| ResNet1D | `src/ecgllava/models/resnet1d.py:44` |
| Classification | `src/ecgllava/models/resnet1d.py:103`; `src/ecgllava/utils/metrics.py:14` |
| 512 × 32 interface | `src/ecgllava/models/resnet1d.py:99`; `results/resnet1d_superdiag_base/metrics.json` → `feature_shape` |

## 02_class_auroc.png / 02_class_auroc.svg

Notion 위치: `Evidence`의 핵심 KPI 바로 아래.

Caption: 공식 test fold 2,158건에서 클래스별 AUROC는 NORM 0.9347, MI 0.9022, STTC 0.9311, CD 0.9080, HYP 0.8956이며, 5-class macro AUROC는 0.9143이다. 고정 threshold 0.5에서 macro F1은 0.7194다.

Claim boundary: `results/resnet1d_superdiag_base/metrics.json`의 `test` 값만 시각화했다. 다른 데이터셋, 반복 run, confidence interval 또는 다른 모델과의 비교가 아니다.

## 03_token_interface.svg / 03_token_interface.png

Notion 위치: `Scope / Limitation` 앞 또는 기술 상세 toggle 내부.

Caption: 10초 12-lead 입력 `(B,12,1000)`을 ResNet1D가 pooling 전 `(B,512,32)`로 변환하며, 시간축 32 step을 temporal token contract로 노출한다. 점선 오른쪽의 projector는 Phase 2 경계이며 아직 구현하지 않았다.

Claim boundary: Phase 1에서 검증된 것은 encoder feature shape까지다. Projector 출력 shape, LLM 연결, 생성 품질 또는 end-to-end 성능을 주장하지 않는다.
