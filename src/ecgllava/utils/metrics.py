"""멀티라벨 평가 지표.

주 지표는 macro AUROC. 클래스별로 AUROC 를 따로 구한 뒤 단순 평균한다.
검증 fold 에 어떤 클래스의 양성이 하나도 없으면 그 클래스의 AUROC 는 정의되지 않으므로
평균에서 제외하고 그 사실을 기록한다.
"""

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

from src.ecgllava.data.labels import SUPERDIAG_CLASSES


def multilabel_metrics(y_true: np.ndarray, y_score: np.ndarray,
                       threshold: float = 0.5) -> dict:
    per_class_auc = {}
    valid = []
    for j, name in enumerate(SUPERDIAG_CLASSES):
        yt = y_true[:, j]
        if yt.min() == yt.max():  # 한 클래스만 존재하면 AUROC 정의 불가
            per_class_auc[name] = None
            continue
        auc = float(roc_auc_score(yt, y_score[:, j]))
        per_class_auc[name] = auc
        valid.append(auc)

    y_pred = (y_score >= threshold).astype(np.int8)
    return {
        "macro_auroc": float(np.mean(valid)) if valid else float("nan"),
        "per_class_auroc": per_class_auc,
        "n_classes_in_macro": len(valid),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "threshold": threshold,
        "n_samples": int(len(y_true)),
    }


def format_metrics(m: dict) -> str:
    per = "  ".join(
        f"{k}={v:.4f}" if v is not None else f"{k}=n/a"
        for k, v in m["per_class_auroc"].items()
    )
    return (f"macroAUROC={m['macro_auroc']:.4f}  macroF1={m['macro_f1']:.4f}  | {per}")
