# ECG-LLaVA 설계 명세

작성일: 2026-08-11
범위: 1단계 (PTB-XL 분류 베이스라인) + 1단계 마무리 (INT8 + Orin 지연)
2단계 (프로젝터 + LoRA SFT) 는 별도 문서에서 다룬다.

---

## 0. 실측 하드웨어와 그로부터 나온 제약

| 항목 | 값 |
|---|---|
| CPU | Intel i7-8750H, 6C/12T |
| RAM | 31GB (가용 19GB) |
| GPU | RTX 2060 Mobile 6GB, sm_75 (Turing) |
| 드라이버 | 535.230.02, CUDA 12.2 runtime 지원 |
| NVMe | Samsung 980 PRO 1TB, `/`, 439GB 여유 |
| HDD | HGST 5400rpm 1TB, `/mnt/hdd_storage`, 725GB 여유 |
| Python | pyenv 3.11.9 (system 3.8.10 은 사용하지 않음) |

여기서 파생되는 설계 제약:

1. **bf16 사실상 불가.** Turing 에는 bf16 텐서코어가 없다. `torch.cuda.is_bf16_supported()` 는 True 를 반환하지만 이는 에뮬레이션을 포함한 값이고, `including_emulation=False` 로 물으면 False 다. 실측 결과 bf16 은 fp16 대비 8.3배 느리고 fp32 보다도 느리다.

   ```
   RTX 2060 sm_75, 2048x2048 matmul, torch 2.5.1+cu121
   fp32   4.55 ms    3.78 TFLOPS
   fp16   0.77 ms   22.25 TFLOPS   <- 사용
   bf16   6.38 ms    2.69 TFLOPS   <- 에뮬레이션
   ```

   혼합정밀도는 fp16 + `torch.amp.GradScaler` 로 고정한다. 예제 코드에서 `bf16=True` 를 보면 전부 `fp16=True` 로 바꾼다. `is_bf16_supported()` 가 True 라고 해서 켜면 조용히 8배 느려진다.
2. **FlashAttention-2 불가.** FA2 는 sm_80 이상 요구. 2단계 LLM 은 `attn_implementation="sdpa"` 로 간다. 설치 시도 금지.
3. **VRAM 6GB 확정.** 2단계 LLM 은 Qwen2.5-0.5B 로 확정한다. 1.5B 는 fp16 가중치만 3.1GB 라 4bit QLoRA 없이는 불가하므로 마감 후 스트레치로 미룬다.
4. **HDD 는 학습 경로에서 제외.** 원본 아카이브만 HDD, 학습이 읽는 전처리 캐시는 NVMe. 100Hz 전량이 fp32 로 약 1.05GB 라 RAM 상주도 가능하다.
5. **DataLoader `num_workers` 상한 6.** 물리 코어 수 기준.

---

## 1. 태스크 정의

- 입력: 12-lead ECG, 10초, 100Hz → 텐서 `(12, 1000)`, float32
- 출력: superdiagnostic 5클래스 **멀티라벨** 로짓 `(5,)`
  - `NORM` 정상, `MI` 심근경색, `STTC` ST/T 변화, `CD` 전도장애, `HYP` 비대
- 손실: `BCEWithLogitsLoss` (한 심전도에 복수 진단이 동시에 붙으므로 softmax 가 아니다)
- 주 지표: **macro AUROC** (클래스별 AUROC 의 단순 평균)
- 보조 지표: 클래스별 AUROC, macro F1 (임계값 0.5), 클래스별 유병률
- 비교 기준선: Strodthoff et al. 2020 벤치마크의 superdiagnostic macro AUC 약 0.92 ~ 0.93

멀티라벨이므로 `sigmoid` 를 클래스마다 독립 적용한다. AUROC 도 클래스별로 따로 계산한 뒤 평균한다.

### 1.1 실측 라벨 통계 (2026-08-11, v1.0.3 메타데이터 기준)

`scp_statements.csv` 에서 `diagnostic == 1` 인 문장 44개가 5개 superdiagnostic 클래스로 매핑된다.

- 전체 레코드 21799, 환자 18869
- superdiagnostic 라벨이 비어 제외되는 레코드 411개 (1.89%) → **사용 레코드 21388**
- 분할: train(fold 1-8) 17084 / val(fold 9) 2146 / test(fold 10) 2158

| 클래스 | 양성 수 | 유병률 |
|---|---|---|
| NORM | 9514 | 44.48% |
| MI | 5469 | 25.57% |
| STTC | 5235 | 24.48% |
| CD | 4898 | 22.90% |
| HYP | 2649 | 12.39% |

레코드당 라벨 개수: 1개 16244, 2개 4068, 3개 919, 4개 157. 전체의 24%가 복수 라벨이므로 멀티라벨 설정이 맞다.

불균형은 최악(HYP 12.4%)이라도 심하지 않다. `pos_weight` 는 베이스라인 AUC 를 본 뒤에 판단한다.

주의: v1.0.1 기준 논문들은 21837 레코드 / superdiagnostic 21430 을 보고한다. v1.0.3 은 21799 / 21388 이다. README 에 버전을 명시해야 수치 비교 시 오해가 없다.

---

## 2. 데이터 파이프라인

### 2.1 경로 규약

```
/mnt/hdd_storage/datasets/ptb-xl/          # 원본 아카이브 (재다운로드 방지용, 읽기 전용 취급)
  ptb-xl-1.0.3.zip
  ptb-xl-1.0.3/                            # 압축 해제본
    ptbxl_database.csv
    scp_statements.csv
    records100/00000/00001_lr.dat|.hea ...
    records500/00000/00001_hr.dat|.hea ...

<repo>/data/cache/                         # NVMe, .gitignore 대상
  ptbxl_100hz.npy                          # (N, 12, 1000) float32
  ptbxl_meta.csv                           # ecg_id, patient_id, strat_fold, y_NORM..y_HYP
  scaler.json                              # lead별 mean/std (학습 fold 에서만 산출)
  prepare_manifest.json                    # 원본 sha256, 생성일시, 레코드 수, 드롭 사유별 카운트
```

`data/cache/` 는 `prepare_ptbxl.py` 로 언제든 재생성 가능하므로 git 에 넣지 않는다. `prepare_manifest.json` 만 `results/` 에 복사해 재현성 근거로 남긴다.

### 2.2 전처리 명세 (`src/ecgllava/data/prepare_ptbxl.py`)

원본 WFDB 포맷(`.dat` + `.hea`) 을 읽어 단일 배열로 만드는 1회성 스크립트다. **`wfdb` 패키지는 이 스크립트에서만 사용하고 학습/추론 경로에는 넣지 않는다.** Orin 배포 시 전처리를 재현해야 하는데 라이브러리 의존을 배포 경로에 끌고 가면 곤란하기 때문이다.

절차:
1. `ptbxl_database.csv` 로드. `ecg_id` 오름차순 고정.
2. `scp_codes` 문자열을 `ast.literal_eval` 로 dict 파싱 (형식: `{'NORM': 100.0, 'SR': 0.0}`)
3. `scp_statements.csv` 에서 `diagnostic == 1` 인 행만 남기고 `diagnostic_class` 컬럼으로 SCP 코드 → superdiagnostic 매핑 테이블 생성
4. 각 레코드의 scp 코드 집합을 매핑해 5차원 multi-hot 벡터 생성. **likelihood 값으로 필터링하지 않는다** (벤치마크 관행과 일치시키기 위함)
5. multi-hot 이 전부 0 인 레코드는 제외하고 그 개수를 manifest 에 기록
6. `records100` 의 `filename_lr` 경로로 신호 읽기 → `(1000, 12)` 를 전치해 `(12, 1000)`
7. `np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)` 적용. NaN 발생 레코드 수를 manifest 에 기록
8. `(N, 12, 1000) float32` 로 `ptbxl_100hz.npy` 저장, 메타는 `ptbxl_meta.csv`

주의: 리드 순서는 헤더 기준 `I, II, III, aVR, aVL, aVF, V1..V6` 로 고정하고 manifest 에 명시한다. 배포 시 입력 리드 순서가 어긋나면 조용히 성능만 떨어진다.

### 2.3 정규화

lead 별(채널 별) 표준화. 통계는 **학습 fold(1-8) 에서만** 산출해 `scaler.json` 에 저장하고, 검증/테스트/배포에 동일 상수를 적용한다. 전체 데이터로 통계를 내면 테스트 정보가 새어 들어간다.

```
x_norm[c, t] = (x[c, t] - mean[c]) / (std[c] + 1e-8)
```

상수 12쌍이므로 Orin 배포 시 전처리에 그대로 하드코딩하면 된다.

### 2.4 분할

`ptbxl_database.csv` 의 `strat_fold` 컬럼을 그대로 쓴다. 환자 단위로 층화되어 있어 직접 나누면 동일 환자가 학습/테스트에 걸쳐 누수된다.

- train: fold 1-8
- val: fold 9 (모델 선택, early stopping)
- test: fold 10 (최종 1회 측정, 그 외에는 열지 않는다)

### 2.5 Dataset / DataLoader (`src/ecgllava/data/dataset.py`)

- `.npy` 를 `mmap_mode='r'` 로 열어 fold 인덱스만 슬라이스
- `__getitem__` 에서 float32 변환 + 정규화 적용
- 1단계 증강은 없음. 베이스라인 숫자를 먼저 확정하고, 개선이 필요하면 그때 추가한다
- `num_workers=6`, `pin_memory=True`, `persistent_workers=True`

---

## 3. 모델 명세 (`src/ecgllava/models/resnet1d.py`)

배포 제약상 **Conv1d, BatchNorm1d, ReLU, MaxPool1d, AdaptiveAvgPool1d, Dropout, Linear** 만 사용한다. RNN 계열, 커스텀 활성화, dynamic shape 유발 연산은 배제한다. ONNX opset 17 로 export 했을 때 TensorRT 가 곧바로 먹는 그래프를 목표로 한다.

```
입력 (B, 12, 1000)

Stem     Conv1d(12, 64, k=15, s=2, p=7) - BN - ReLU      -> (B, 64, 500)
         MaxPool1d(k=3, s=2, p=1)                        -> (B, 64, 250)

Stage1   BasicBlock1d(64,  64,  stride=1) x2             -> (B, 64,  250)
Stage2   BasicBlock1d(64,  128, stride=2) + x1           -> (B, 128, 125)
Stage3   BasicBlock1d(128, 256, stride=2) + x1           -> (B, 256, 63)
Stage4   BasicBlock1d(256, 512, stride=2) + x1           -> (B, 512, 32)

Head     AdaptiveAvgPool1d(1) - Flatten                  -> (B, 512)
         Dropout(p=0.5) - Linear(512, 5)                 -> (B, 5)
```

`BasicBlock1d(cin, cout, stride)`:
```
out = Conv1d(cin, cout, k=7, s=stride, p=3) - BN - ReLU
out = Conv1d(cout, cout, k=7, s=1, p=3) - BN
skip = identity            (cin == cout and stride == 1)
     = Conv1d(cin, cout, k=1, s=stride) - BN    (그 외)
return ReLU(out + skip)
```

파라미터 약 8.6M. 6GB VRAM 에서 batch 128 까지 여유롭다.

**2단계와의 연결**: 이 인코더의 `AdaptiveAvgPool1d` 직전 특징맵 `(B, 512, 32)` 가 2단계 프로젝터의 입력이 된다. LLaVA 에서 CLIP ViT 의 패치 토큰 시퀀스가 프로젝터를 거쳐 LLM 토큰 공간으로 들어가는 것과 같은 자리다. 시간축 32 스텝이 곧 32개의 "ECG 토큰"이 된다. 따라서 1단계 체크포인트는 2단계에서 그대로 재사용하며, 최종 pooling 과 분류 헤드만 떼어낸다.

---

## 4. 학습 명세 (`src/ecgllava/train.py`)

| 항목 | 값 | 근거 |
|---|---|---|
| optimizer | AdamW, weight_decay 1e-2 | |
| lr | 1e-3 | batch 128 기준. 발산 시 3e-4 |
| scheduler | CosineAnnealingLR, warmup 3 epoch | |
| epochs | 50, early stopping patience 10 (val macro AUROC) | |
| batch_size | 128 | VRAM 여유 확인 후 조정 |
| AMP | fp16 + GradScaler | Turing 은 bf16 불가 |
| grad clip | 1.0 | fp16 안정성 |
| seed | 42, `torch.backends.cudnn.deterministic=True` | 재현성 |
| 체크포인트 | val macro AUROC 최고 시점만 저장 | |

fp16 특유의 실패 모드: loss 가 NaN 이 되면 GradScaler 의 스케일이 계속 반감된다. 학습 로그에 `scaler.get_scale()` 을 남겨 조기에 감지한다.

---

## 5. 평가와 결과 기록 (`src/ecgllava/eval.py`)

실험 1회당 `results/<exp_name>/` 아래에 다음을 남긴다. 하나라도 빠지면 그 실험은 미완으로 본다.

```
results/<exp_name>/
  config.json          # 실행에 쓰인 config dataclass 전체 덤프
  env.json             # torch/cuda/driver 버전, GPU 이름, git commit hash
  prepare_manifest.json  # 데이터 캐시 생성 근거 사본
  metrics.json         # val/test macro AUROC, 클래스별 AUROC, macro F1, epoch 수
  curves.csv           # epoch, train_loss, val_loss, val_macro_auroc, lr, grad_scale
  best.pt              # state_dict only
```

테스트 fold 는 학습이 완전히 끝난 뒤 1회만 평가한다.

---

## 6. Config 명세 (`configs/`)

dataclass 기반. 의존성 없음, 타입 체크 동작, IDE 자동완성 동작.

```python
# configs/base.py
from dataclasses import dataclass, field, asdict

@dataclass
class DataCfg:
    cache_dir: str = "data/cache"
    raw_root: str = "/mnt/hdd_storage/datasets/ptb-xl/ptb-xl-1.0.3"
    sampling_rate: int = 100          # 100 | 500
    label_set: str = "superdiagnostic"
    train_folds: tuple = (1, 2, 3, 4, 5, 6, 7, 8)
    val_folds: tuple = (9,)
    test_folds: tuple = (10,)
    num_workers: int = 6

@dataclass
class ModelCfg:
    name: str = "resnet1d"
    in_channels: int = 12
    stem_channels: int = 64
    stage_channels: tuple = (64, 128, 256, 512)
    blocks_per_stage: tuple = (2, 2, 2, 2)
    kernel_size: int = 7
    dropout: float = 0.5
    num_classes: int = 5

@dataclass
class TrainCfg:
    seed: int = 42
    epochs: int = 50
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-2
    warmup_epochs: int = 3
    grad_clip: float = 1.0
    amp: bool = True                  # fp16 고정, bf16 미지원 하드웨어
    early_stop_patience: int = 10

@dataclass
class Cfg:
    exp_name: str = "resnet1d_superdiag_base"
    out_dir: str = "results"
    data: DataCfg = field(default_factory=DataCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
```

실험별 config 는 `configs/exp_*.py` 에서 `Cfg()` 를 만든 뒤 필드만 덮어쓴다. 실행 시 `--config configs/exp_xxx.py` 로 경로를 받아 `cfg` 심볼을 import 한다. CLI 오버라이드는 넣지 않는다. 모든 변경은 파일에 남아야 재현이 된다.

---

## 7. 레포 구조

```
ECG-LLaVA/
  CLAUDE.md
  DESIGN.md
  README.md                  # 최종 수치를 기록하는 곳
  requirements.txt
  .gitignore                 # data/, results/*/best.pt, *.npy
  configs/
    base.py
    exp_resnet1d_superdiag.py
  src/ecgllava/
    __init__.py
    data/
      prepare_ptbxl.py
      labels.py
      dataset.py
    models/
      resnet1d.py
    utils/
      seed.py
      metrics.py
      logging.py
    train.py
    eval.py
    export_onnx.py           # 1단계 마무리 단계에서 추가
  data/cache/                # gitignore
  results/
```

---

## 8. 환경 구축

```bash
~/.pyenv/versions/3.11.9/bin/python -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu121
.venv/bin/pip install numpy pandas scikit-learn wfdb tqdm
```

- Python 3.11.9 (pyenv). system 3.8 은 최신 transformers 미지원이라 2단계에서 막힌다.
- cu121 휠 선택 근거: 드라이버 535 는 CUDA 12.2 까지 지원. cu124 도 minor version compatibility 로 동작하지만 안전한 쪽을 택한다.
- `wfdb` 는 전처리 전용. 학습/추론 경로에 import 하지 않는다.

---

## 9. 일정

| 날짜 | 항목 | 산출물 |
|---|---|---|
| 8/11 | PTB-XL 다운로드, 전처리 스크립트, 캐시 생성 | `ptbxl_100hz.npy`, `prepare_manifest.json` |
| 8/12-13 | dataset, model, train loop, 1 epoch smoke test | 학습이 도는 상태 |
| 8/14-15 | 전체 학습, val/test 평가 | `results/resnet1d_superdiag_base/` |
| 8/16-17 | README 수치 기록, 재현 절차 검증 | 1단계 완료 |
| 8/18-21 | ONNX export, TensorRT INT8 캘리브레이션 | `export_onnx.py`, engine |
| 8/22-24 | Orin 지연 측정, README 기록 | 배포 수치 |
| 8/25-31 | (스트레치) 프로젝터 + LoRA SFT | 2단계 설계 문서 별도 |

절단 규칙: 밀리면 2단계부터 자른다. 1단계 AUC 와 Orin 배포 수치는 불가침.

---

## 10. 미확정 항목

해결됨:
- ~~실제 레코드 수와 제외 개수~~ → 1.1 절에 실측 반영 (21799 → 21388)
- ~~batch_size 128 이 6GB 에 들어가는가~~ → 최대 VRAM 0.36GB. 한참 여유다.
- ~~NaN 레코드 수~~ → 0건. 리드 순서 불일치 0건, 길이 불일치 0건.

남은 판단:
- 클래스 불균형 대응(pos_weight). 베이스라인이 나왔으므로 이제 판단 가능하다. 12절 참조.

---

## 12. 1단계 베이스라인 결과와 다음 판단

`results/resnet1d_superdiag_base/` (2026-08-11)

| | val (fold 9) | test (fold 10) |
|---|---|---|
| macro AUROC | 0.9203 (epoch 5) | **0.9143** |
| macro F1 | 0.7143 | 0.7194 |

클래스별 test AUROC: NORM 0.9347, MI 0.9022, STTC 0.9311, CD 0.9080, HYP 0.8956

epoch 당 4.8초, 최대 VRAM 0.36GB, early stopping 으로 15 epoch 에서 종료.
`eval.py` 로 체크포인트를 다시 불러 평가해도 0.9143 이 그대로 재현된다.

**관찰: 과적합이 명확하다.** train loss 는 0.416 -> 0.136 으로 계속 내려가는데
val loss 는 epoch 9 (0.2905) 를 바닥으로 다시 올라간다. val AUROC 는 epoch 5 이후 정체다.
용량 부족이 아니라 정규화 부족이다. 다음 중에서 고른다.

1. 증강 추가 (random crop, 시간축 이동, lead dropout, 가우시안 노이즈)
2. weight decay 상향 (1e-2 -> 5e-2), dropout 상향
3. 모델 축소 (blocks_per_stage 2,2,2,2 -> 1,1,1,1)

VRAM 0.36GB, epoch 4.8초라 실험 1회가 1.5분이다. 탐색 비용이 사실상 없다.
다만 CLAUDE.md 의 "과욕 금지" 원칙에 따라 베이스라인 수치를 먼저 확정해 기록했고,
개선 여부는 별도 판단으로 남긴다. pos_weight 도 이 실험군에 함께 넣는다.

---

## 11. 데이터 취득 기록

PhysioNet 직접 다운로드(`physionet.org/content/ptb-xl/get-zip/1.0.3/`)는 실측 0.2MB/s 로
1.84GB zip 에 2시간 이상 걸렸다. PhysioNet 이 AWS Open Data 로 공개하는 미러를 대신 사용한다.

```
https://physionet-open.s3.amazonaws.com/ptb-xl/1.0.3/
```

자격증명 불필요, 오브젝트 87204개, 총 3.18GB (records100 43598, records500 43598, 메타 8).
파일 단위라 요청 지연이 지배적이므로 48 병렬로 받는다. 실측 58 파일/초.

수집 스크립트는 `tools/fetch_ptbxl.sh` 로 레포에 보존한다. 이미 존재하는 파일은 건너뛰므로
중단 후 재실행이 안전하다. 무결성은 `SHA256SUMS.txt` 로 검증한다.

수집 결과 (2026-08-11):

| 대상 | 파일 수 | 용량 | 검증 |
|---|---|---|---|
| 메타데이터 | 8 | 16MB | ok |
| records100 | 43598 | 613MB | ok |
| records500 | 43598 | 2.6GB | ok |
| 합계 | 87204 | 3.2GB | **sha256 87203/87203 일치, 불일치 0** |

(`SHA256SUMS.txt` 자신은 목록에 포함되지 않으므로 검증 항목이 오브젝트 수보다 1 적다.)

재현 절차:

```bash
python3 tools/list_ptbxl_s3.py            # keys_all.txt 생성
tools/fetch_ptbxl.sh keys_all.txt 48      # 병렬 수집
cd /mnt/hdd_storage/datasets/ptb-xl/ptb-xl-1.0.3 && sha256sum -c SHA256SUMS.txt
```
