"""PTB-XL 원본(WFDB) -> 학습용 캐시 생성. 1회성 스크립트.

wfdb 패키지는 이 파일에서만 쓴다. 학습/추론 경로에는 넣지 않는다.
Orin 배포 시 전처리를 재현해야 하는데 라이브러리 의존을 배포 경로에 끌고 가면 곤란하기 때문.

산출물 (DESIGN.md 2.1절):
  <cache>/ptbxl_{rate}hz.npy      (N, 12, L) float32
  <cache>/ptbxl_meta.csv          ecg_id, patient_id, strat_fold, y_NORM..y_HYP
  <cache>/scaler.json             lead별 mean/std (학습 fold 에서만 산출)
  <cache>/prepare_manifest.json   재현 근거

사용법:
  python -m src.ecgllava.data.prepare_ptbxl --config configs/exp_resnet1d_superdiag.py
"""

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from src.ecgllava.data.labels import (  # noqa: E402
    SUPERDIAG_CLASSES,
    build_label_matrix,
    load_scp_to_superdiag,
)
from src.ecgllava.utils.config import load_cfg  # noqa: E402

# 헤더 기준 표준 리드 순서. 배포 시 입력 순서가 어긋나면 조용히 성능만 떨어진다.
EXPECTED_LEADS = ["I", "II", "III", "AVR", "AVL", "AVF",
                  "V1", "V2", "V3", "V4", "V5", "V6"]

_RAW_ROOT = None


def _init_worker(raw_root: str) -> None:
    global _RAW_ROOT
    _RAW_ROOT = raw_root


def _read_one(rel_path: str):
    """WFDB 레코드 하나를 (12, L) float32 로 읽는다. 리드 순서 검증 결과를 함께 반환."""
    sig, fields = wfdb.rdsamp(os.path.join(_RAW_ROOT, rel_path))
    leads = [s.strip().upper() for s in fields["sig_name"]]
    x = np.asarray(sig, dtype=np.float32).T  # (n_samples, 12) -> (12, n_samples)
    n_nan = int(np.isnan(x).sum())
    if n_nan:
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x, leads == EXPECTED_LEADS, n_nan


def sha256_of(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    raw_root = cfg.data.raw_root
    cache_dir = cfg.data.cache_dir
    rate = cfg.data.sampling_rate
    os.makedirs(cache_dir, exist_ok=True)

    db_path = os.path.join(raw_root, "ptbxl_database.csv")
    db = pd.read_csv(db_path, index_col="ecg_id").sort_index()
    n_total = len(db)

    # 1) 라벨 생성. superdiagnostic 이 비는 레코드는 제외한다.
    mapping = load_scp_to_superdiag(raw_root)
    y_all = build_label_matrix(db, mapping)
    keep = y_all.sum(axis=1) > 0
    n_empty = int((~keep).sum())
    db = db[keep]
    y = y_all[keep]
    print(f"[label] 전체 {n_total}, 라벨 없음 {n_empty} 제외, 사용 {len(db)}")

    # 2) 신호 읽기
    col = "filename_lr" if rate == 100 else "filename_hr"
    rel_paths = db[col].tolist()
    expected_len = rate * 10  # 10초 고정

    signals = np.empty((len(rel_paths), 12, expected_len), dtype=np.float32)
    n_lead_mismatch = 0
    n_nan_records = 0
    n_len_mismatch = 0

    with ProcessPoolExecutor(args.workers, initializer=_init_worker,
                             initargs=(raw_root,)) as ex:
        it = ex.map(_read_one, rel_paths, chunksize=64)
        for i, (x, lead_ok, n_nan) in enumerate(
                tqdm(it, total=len(rel_paths), desc=f"read {rate}Hz")):
            if not lead_ok:
                n_lead_mismatch += 1
            if n_nan:
                n_nan_records += 1
            if x.shape[1] != expected_len:
                n_len_mismatch += 1
                buf = np.zeros((12, expected_len), dtype=np.float32)
                n = min(x.shape[1], expected_len)
                buf[:, :n] = x[:, :n]
                x = buf
            signals[i] = x

    print(f"[signal] shape={signals.shape} dtype={signals.dtype} "
          f"({signals.nbytes / 1e9:.2f} GB)")
    print(f"[signal] 리드 순서 불일치 {n_lead_mismatch}, NaN 포함 {n_nan_records}, "
          f"길이 불일치 {n_len_mismatch}")

    # 3) 정규화 통계는 학습 fold 에서만 산출한다. 전체로 내면 테스트 정보가 샌다.
    folds = db["strat_fold"].to_numpy()
    train_mask = np.isin(folds, cfg.data.train_folds)
    train_sig = signals[train_mask]
    mean = train_sig.mean(axis=(0, 2)).astype(np.float64)
    std = train_sig.std(axis=(0, 2)).astype(np.float64)
    print(f"[scaler] train {int(train_mask.sum())}개 기준, "
          f"mean[0]={mean[0]:.5f} std[0]={std[0]:.5f}")

    # 4) 저장
    sig_path = os.path.join(cache_dir, f"ptbxl_{rate}hz.npy")
    np.save(sig_path, signals)

    meta = pd.DataFrame({
        "ecg_id": db.index.to_numpy(),
        "patient_id": db["patient_id"].to_numpy(),
        "strat_fold": folds,
    })
    for j, c in enumerate(SUPERDIAG_CLASSES):
        meta[f"y_{c}"] = y[:, j].astype(np.int8)
    meta_path = os.path.join(cache_dir, "ptbxl_meta.csv")
    meta.to_csv(meta_path, index=False)

    scaler = {
        "leads": EXPECTED_LEADS,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "eps": 1e-8,
        "computed_on": {"folds": list(cfg.data.train_folds),
                        "n_records": int(train_mask.sum())},
    }
    with open(os.path.join(cache_dir, "scaler.json"), "w") as f:
        json.dump(scaler, f, indent=2)

    manifest = {
        "created_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "raw_root": raw_root,
        "ptbxl_version": "1.0.3",
        "sampling_rate": rate,
        "label_set": cfg.data.label_set,
        "classes": SUPERDIAG_CLASSES,
        "n_total_records": n_total,
        "n_dropped_empty_label": n_empty,
        "n_used": int(len(db)),
        "n_lead_order_mismatch": n_lead_mismatch,
        "n_records_with_nan": n_nan_records,
        "n_length_mismatch": n_len_mismatch,
        "lead_order": EXPECTED_LEADS,
        "signal_shape": list(signals.shape),
        "signal_dtype": str(signals.dtype),
        "split": {
            "train_folds": list(cfg.data.train_folds),
            "val_folds": list(cfg.data.val_folds),
            "test_folds": list(cfg.data.test_folds),
            "n_train": int(np.isin(folds, cfg.data.train_folds).sum()),
            "n_val": int(np.isin(folds, cfg.data.val_folds).sum()),
            "n_test": int(np.isin(folds, cfg.data.test_folds).sum()),
        },
        "class_positives": {c: int(y[:, j].sum())
                            for j, c in enumerate(SUPERDIAG_CLASSES)},
        "sha256": {
            "ptbxl_database.csv": sha256_of(db_path),
            "scp_statements.csv": sha256_of(
                os.path.join(raw_root, "scp_statements.csv")),
        },
    }
    with open(os.path.join(cache_dir, "prepare_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"[done] {sig_path}")
    print(f"[done] {meta_path}")
    print(f"[done] split train={manifest['split']['n_train']} "
          f"val={manifest['split']['n_val']} test={manifest['split']['n_test']}")


if __name__ == "__main__":
    main()
