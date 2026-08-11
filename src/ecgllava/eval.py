"""저장된 체크포인트로 임의 split 을 재평가한다.

학습과 분리해 두는 이유: 재현성 검증(같은 체크포인트가 같은 수치를 내는가)과
2단계에서 인코더를 불러올 때 동일한 로딩 경로를 쓰기 위함이다.

사용법:
  python -m src.ecgllava.eval --config configs/exp_resnet1d_superdiag.py --split test
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.ecgllava.data.dataset import build_loaders  # noqa: E402
from src.ecgllava.models.resnet1d import build_model  # noqa: E402
from src.ecgllava.utils.config import load_cfg  # noqa: E402
from src.ecgllava.utils.metrics import format_metrics, multilabel_metrics  # noqa: E402
from src.ecgllava.utils.seed import set_seed  # noqa: E402


@torch.no_grad()
def run(model, loader, device, amp: bool):
    model.eval()
    scores, targets = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            logits = model(x)
        scores.append(torch.sigmoid(logits.float()).cpu().numpy())
        targets.append(y.numpy())
    return np.concatenate(targets), np.concatenate(scores)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--ckpt", default=None, help="기본값: results/<exp_name>/best.pt")
    ap.add_argument("--save-scores", action="store_true",
                    help="예측 확률을 npy 로 저장 (임계값 분석용)")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    set_seed(cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.join(cfg.out_dir, cfg.exp_name)
    ckpt = args.ckpt or os.path.join(out_dir, "best.pt")

    _, loaders = build_loaders(cfg)
    model = build_model(cfg.model).to(device)
    model.load_state_dict(torch.load(ckpt, weights_only=True))
    print(f"[ckpt] {ckpt}")

    y_true, y_score = run(model, loaders[args.split], device, cfg.train.amp)
    m = multilabel_metrics(y_true, y_score)
    print(f"[{args.split}] {format_metrics(m)}")
    print(json.dumps(m, indent=2))

    if args.save_scores:
        np.save(os.path.join(out_dir, f"scores_{args.split}.npy"), y_score)
        np.save(os.path.join(out_dir, f"targets_{args.split}.npy"), y_true)
        print(f"[done] scores/targets 저장: {out_dir}")


if __name__ == "__main__":
    main()
