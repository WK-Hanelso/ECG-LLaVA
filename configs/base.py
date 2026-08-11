"""실험 설정 스키마.

CLI 오버라이드는 두지 않는다. 모든 변경은 파일에 남아야 재현이 된다.
실험별 설정은 configs/exp_*.py 에서 이 dataclass 를 만든 뒤 필드만 덮어쓴다.
"""

from dataclasses import asdict, dataclass, field


@dataclass
class DataCfg:
    # 원본 아카이브 (HDD). 읽기 전용으로만 쓴다.
    raw_root: str = "/mnt/hdd_storage/datasets/ptb-xl/ptb-xl-1.0.3"
    # 전처리 캐시 (NVMe). prepare_ptbxl.py 로 언제든 재생성 가능.
    cache_dir: str = "data/cache"

    sampling_rate: int = 100  # 100 | 500
    label_set: str = "superdiagnostic"

    # PTB-XL 이 제공하는 strat_fold 를 그대로 쓴다. 환자 단위 층화라 직접 나누면 누수된다.
    train_folds: tuple = (1, 2, 3, 4, 5, 6, 7, 8)
    val_folds: tuple = (9,)
    test_folds: tuple = (10,)

    num_workers: int = 6  # 물리 코어 6개 기준
    pin_memory: bool = True


@dataclass
class ModelCfg:
    name: str = "resnet1d"
    in_channels: int = 12
    stem_channels: int = 64
    stem_kernel: int = 15
    stage_channels: tuple = (64, 128, 256, 512)
    blocks_per_stage: tuple = (2, 2, 2, 2)
    stage_strides: tuple = (1, 2, 2, 2)  # 1000 -> 32 스텝. 2단계 ECG 토큰 수와 직결.
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
    amp: bool = True  # fp16 고정. Turing 은 bf16 네이티브 미지원 (DESIGN.md 0절).
    early_stop_patience: int = 10
    pos_weight: bool = False  # 베이스라인 AUC 확인 후 판단


@dataclass
class Cfg:
    exp_name: str = "resnet1d_superdiag_base"
    out_dir: str = "results"
    data: DataCfg = field(default_factory=DataCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    train: TrainCfg = field(default_factory=TrainCfg)

    def to_dict(self) -> dict:
        return asdict(self)
