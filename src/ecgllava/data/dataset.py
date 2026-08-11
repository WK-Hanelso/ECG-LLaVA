"""전처리 캐시를 읽는 Dataset.

.npy 를 mmap 으로 열고 fold 인덱스만 슬라이스한다. 100Hz 전량이 1.0GB 라 OS 페이지 캐시에
그대로 올라간다. 정규화 상수는 학습 fold 에서 산출된 scaler.json 을 그대로 쓴다.

1단계는 증강을 넣지 않는다. 베이스라인 숫자를 먼저 확정한다.
"""

import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.ecgllava.data.labels import SUPERDIAG_CLASSES


class PTBXLDataset(Dataset):
    def __init__(self, cache_dir: str, sampling_rate: int, folds):
        sig_path = os.path.join(cache_dir, f"ptbxl_{sampling_rate}hz.npy")
        meta_path = os.path.join(cache_dir, "ptbxl_meta.csv")
        scaler_path = os.path.join(cache_dir, "scaler.json")
        for p in (sig_path, meta_path, scaler_path):
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"{p} 없음. prepare_ptbxl.py 를 먼저 실행한다.")

        self.signals = np.load(sig_path, mmap_mode="r")
        meta = pd.read_csv(meta_path)
        self.indices = np.where(meta["strat_fold"].isin(list(folds)).to_numpy())[0]

        label_cols = [f"y_{c}" for c in SUPERDIAG_CLASSES]
        self.labels = meta[label_cols].to_numpy(dtype=np.float32)
        self.ecg_ids = meta["ecg_id"].to_numpy()

        with open(scaler_path) as f:
            scaler = json.load(f)
        # (12, 1) 로 두면 브로드캐스트로 lead 별 정규화가 된다.
        self.mean = np.asarray(scaler["mean"], dtype=np.float32)[:, None]
        self.std = np.asarray(scaler["std"], dtype=np.float32)[:, None]
        self.eps = float(scaler["eps"])

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = self.indices[i]
        x = np.asarray(self.signals[idx], dtype=np.float32)
        x = (x - self.mean) / (self.std + self.eps)
        return torch.from_numpy(x), torch.from_numpy(self.labels[idx])

    def pos_weight(self) -> torch.Tensor:
        """클래스별 (음성 수 / 양성 수). BCEWithLogitsLoss 의 pos_weight 용."""
        y = self.labels[self.indices]
        pos = y.sum(axis=0)
        neg = len(y) - pos
        return torch.from_numpy((neg / np.maximum(pos, 1.0)).astype(np.float32))


def build_loaders(cfg):
    ds = {
        "train": PTBXLDataset(cfg.data.cache_dir, cfg.data.sampling_rate,
                              cfg.data.train_folds),
        "val": PTBXLDataset(cfg.data.cache_dir, cfg.data.sampling_rate,
                            cfg.data.val_folds),
        "test": PTBXLDataset(cfg.data.cache_dir, cfg.data.sampling_rate,
                             cfg.data.test_folds),
    }
    common = dict(num_workers=cfg.data.num_workers, pin_memory=cfg.data.pin_memory,
                  persistent_workers=cfg.data.num_workers > 0)
    loaders = {
        "train": DataLoader(ds["train"], batch_size=cfg.train.batch_size,
                            shuffle=True, drop_last=True, **common),
        "val": DataLoader(ds["val"], batch_size=cfg.train.batch_size,
                          shuffle=False, **common),
        "test": DataLoader(ds["test"], batch_size=cfg.train.batch_size,
                           shuffle=False, **common),
    }
    return ds, loaders
