"""SCP 코드 -> superdiagnostic 5클래스 멀티라벨 변환.

PTB-XL 의 라벨은 ptbxl_database.csv 의 scp_codes 컬럼에 {코드: 확신도} dict 문자열로 들어있다.
scp_statements.csv 에서 diagnostic == 1 인 문장만 골라 diagnostic_class 로 상위 매핑한다.
확신도로 필터링하지 않는다 (공개 벤치마크 관행과 일치시키기 위함).
"""

import ast
import os

import numpy as np
import pandas as pd

SUPERDIAG_CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
CLASS_TO_IDX = {c: i for i, c in enumerate(SUPERDIAG_CLASSES)}


def load_scp_to_superdiag(raw_root: str) -> dict:
    """diagnostic 문장 44개에 대한 {scp_code: superdiag_class} 매핑."""
    path = os.path.join(raw_root, "scp_statements.csv")
    df = pd.read_csv(path, index_col=0)
    df = df[df["diagnostic"] == 1]
    mapping = df["diagnostic_class"].dropna().to_dict()
    unknown = set(mapping.values()) - set(SUPERDIAG_CLASSES)
    if unknown:
        raise ValueError(f"예상 외 superdiagnostic 클래스: {sorted(unknown)}")
    return mapping


def parse_scp_codes(raw: str) -> dict:
    return ast.literal_eval(raw)


def to_multihot(scp_codes: dict, mapping: dict) -> np.ndarray:
    """{코드: 확신도} -> (5,) float32 multi-hot. 매핑되는 코드가 없으면 전부 0."""
    y = np.zeros(len(SUPERDIAG_CLASSES), dtype=np.float32)
    for code in scp_codes:
        cls = mapping.get(code)
        if cls is not None:
            y[CLASS_TO_IDX[cls]] = 1.0
    return y


def build_label_matrix(db: pd.DataFrame, mapping: dict) -> np.ndarray:
    """ptbxl_database.csv DataFrame -> (N, 5) float32 라벨 행렬."""
    return np.stack([
        to_multihot(parse_scp_codes(s), mapping) for s in db["scp_codes"]
    ])
