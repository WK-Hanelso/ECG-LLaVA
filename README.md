# ECG-LLaVA

심전도(ECG)를 입력받아 자연어 판독을 생성하는 멀티모달 모델. LLaVA 레시피를 생체신호에 이식한다.

1단계는 PTB-XL 부정맥 분류 베이스라인, 2단계는 프로젝터 + LoRA SFT 로 판독문 생성이다.
설계 명세는 [DESIGN.md](DESIGN.md) 에 있다.

---

## 1단계 결과: PTB-XL superdiagnostic 5클래스 멀티라벨

| 항목 | 값 |
|---|---|
| 데이터 | PTB-XL **v1.0.3**, 100Hz, 12-lead, 10초 |
| 사용 레코드 | 21388 (전체 21799 중 superdiagnostic 라벨 없는 411개 제외) |
| 분할 | 공식 `strat_fold` 기준 train 1-8 (17084) / val 9 (2146) / test 10 (2158) |
| 모델 | ResNet1d, 8.74M params |
| 학습 | AdamW, lr 1e-3, cosine + warmup 3, fp16 AMP, batch 128 |
| 학습 시간 | epoch 당 4.8초, early stop 으로 15 epoch 종료 (RTX 2060 6GB) |
| 최대 VRAM | 0.36 GB |

### test fold (fold 10) 성능

| 지표 | 값 |
|---|---|
| **macro AUROC** | **0.9143** |
| macro F1 (th=0.5) | 0.7194 |

클래스별 AUROC:

| NORM | MI | STTC | CD | HYP |
|---|---|---|---|---|
| 0.9347 | 0.9022 | 0.9311 | 0.9080 | 0.8956 |

val fold 최고 성능은 epoch 5 에서 macro AUROC 0.9203.

> 수치 비교 시 주의: 널리 인용되는 Strodthoff et al. 2020 벤치마크(superdiagnostic macro AUC 약 0.93)는
> PTB-XL **v1.0.1** (21837 레코드, superdiagnostic 21430) 기준이다.
> 이 저장소는 **v1.0.3** (21799 / 21388) 을 쓴다. 데이터셋 버전이 다르므로 직접 비교는 근사치로만 본다.

---

## 재현 절차

```bash
# 1) 환경
~/.pyenv/versions/3.11.9/bin/python -m venv .venv
.venv/bin/pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
.venv/bin/pip install -r requirements.txt

# 2) 데이터 (PhysioNet AWS 공개 미러, 자격증명 불필요)
python tools/list_ptbxl_s3.py                 # keys_all.txt 생성
tools/fetch_ptbxl.sh keys_all.txt 48          # 87204 파일, 3.2GB
cd /mnt/hdd_storage/datasets/ptb-xl/ptb-xl-1.0.3 && sha256sum -c SHA256SUMS.txt

# 3) 전처리 (약 30초)
.venv/bin/python -m src.ecgllava.data.prepare_ptbxl \
    --config configs/exp_resnet1d_superdiag.py

# 4) 학습 (약 1.5분)
.venv/bin/python -m src.ecgllava.train --config configs/exp_resnet1d_superdiag.py

# 5) 재평가
.venv/bin/python -m src.ecgllava.eval \
    --config configs/exp_resnet1d_superdiag.py --split test
```

시드는 42 로 고정하고 cuDNN 결정론 옵션을 켠다. 실험 1회당
`results/<exp_name>/` 에 `config.json`, `env.json`, `prepare_manifest.json`,
`metrics.json`, `curves.csv`, `best.pt` 가 남는다.

---

## 구조

```
configs/            dataclass 기반 설정. CLI 오버라이드 없음 (모든 변경이 파일에 남아야 재현된다)
src/ecgllava/
  data/prepare_ptbxl.py   WFDB 원본 -> npy 캐시. wfdb 의존은 여기서 끝난다
  data/labels.py          SCP 코드 -> superdiagnostic 5클래스 multi-hot
  data/dataset.py         mmap 캐시 로딩 + lead별 정규화
  models/resnet1d.py      1D ResNet 인코더 + 분류 헤드
  train.py / eval.py
tools/              데이터 수집 및 통계 스크립트
results/            실험별 config / metrics / curves
```

인코더의 `forward_features` 는 `(B, 512, 32)` 를 반환한다. 시간축 32 스텝이 곧 32개의
ECG 토큰이고, 2단계에서 프로젝터를 거쳐 LLM 임베딩 공간으로 들어간다.
LLaVA 에서 CLIP 패치 토큰이 놓이는 자리와 같다.

배포를 염두에 두고 Conv1d / BatchNorm1d / ReLU / Pool / Linear 만 사용한다.
RNN 계열과 커스텀 활성화는 배제해 ONNX -> TensorRT 변환 경로를 단순하게 유지한다.

---

## 진행 상황

- [x] 1단계: PTB-XL 분류 베이스라인, test macro AUROC 0.9143
- [ ] 1단계 마무리: INT8 양자화 + Jetson AGX Orin 지연 측정
- [ ] 2단계: 프로젝터 + LoRA SFT 판독문 생성 (스트레치)
