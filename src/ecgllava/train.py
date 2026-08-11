"""1단계 학습 루프.

사용법:
  python -m src.ecgllava.train --config configs/exp_resnet1d_superdiag.py

산출물 (DESIGN.md 5절): results/<exp_name>/ 아래 config.json, env.json, metrics.json,
curves.csv, best.pt. 하나라도 빠지면 그 실험은 미완으로 본다.

정밀도는 fp16 AMP 고정이다. Turing 은 bf16 텐서코어가 없어 bf16 은 fp16 대비 8.3배 느리다.
fp16 은 loss scale 관리가 필요하므로 curves.csv 에 grad_scale 을 남겨 발산을 조기 감지한다.
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.ecgllava.data.dataset import build_loaders  # noqa: E402
from src.ecgllava.models.resnet1d import build_model  # noqa: E402
from src.ecgllava.utils.config import load_cfg  # noqa: E402
from src.ecgllava.utils.metrics import format_metrics, multilabel_metrics  # noqa: E402
from src.ecgllava.utils.seed import set_seed  # noqa: E402


def collect_env() -> dict:
    env = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        env["compute_capability"] = f"{p.major}.{p.minor}"
        env["gpu_memory_gb"] = round(p.total_memory / 1e9, 2)
        env["bf16_native"] = torch.cuda.is_bf16_supported(including_emulation=False)
    try:
        env["driver"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True).strip()
    except Exception:
        env["driver"] = None
    try:
        env["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        env["git_commit"] = None
    return env


def build_scheduler(optimizer, cfg, steps_per_epoch: int):
    """epoch 단위 linear warmup 후 cosine decay."""
    warmup = cfg.train.warmup_epochs
    total = cfg.train.epochs

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup:
            return (epoch + 1) / max(warmup, 1)
        progress = (epoch - warmup) / max(total - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate(model, loader, criterion, device, amp: bool) -> tuple:
    model.eval()
    losses, scores, targets = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            logits = model(x)
            loss = criterion(logits, y)
        losses.append(loss.item() * len(x))
        scores.append(torch.sigmoid(logits.float()).cpu().numpy())
        targets.append(y.cpu().numpy())
    n = sum(len(t) for t in targets)
    metrics = multilabel_metrics(np.concatenate(targets), np.concatenate(scores))
    return sum(losses) / n, metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--epochs", type=int, default=None,
                    help="smoke test 전용. 실험 기록에는 config 값을 쓴다.")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
        cfg.exp_name = f"{cfg.exp_name}_smoke{args.epochs}"

    set_seed(cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.join(cfg.out_dir, cfg.exp_name)
    os.makedirs(out_dir, exist_ok=True)

    ds, loaders = build_loaders(cfg)
    print(f"[data] train={len(ds['train'])} val={len(ds['val'])} test={len(ds['test'])}")

    model = build_model(cfg.model).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {cfg.model.name}, params={n_params/1e6:.2f}M")
    with torch.no_grad():
        feat = model.forward_features(torch.zeros(1, cfg.model.in_channels,
                                                  cfg.data.sampling_rate * 10,
                                                  device=device))
    print(f"[model] forward_features -> {tuple(feat.shape)} "
          f"(2단계 ECG 토큰 {feat.shape[-1]}개)")

    pos_weight = ds["train"].pos_weight().to(device) if cfg.train.pos_weight else None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                                  weight_decay=cfg.train.weight_decay)
    scheduler = build_scheduler(optimizer, cfg, len(loaders["train"]))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.train.amp)

    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg.to_dict(), f, indent=2, default=str)
    with open(os.path.join(out_dir, "env.json"), "w") as f:
        json.dump(collect_env(), f, indent=2)
    manifest_src = os.path.join(cfg.data.cache_dir, "prepare_manifest.json")
    if os.path.exists(manifest_src):
        with open(manifest_src) as f, \
                open(os.path.join(out_dir, "prepare_manifest.json"), "w") as g:
            g.write(f.read())

    curves_path = os.path.join(out_dir, "curves.csv")
    with open(curves_path, "w") as f:
        f.write("epoch,train_loss,val_loss,val_macro_auroc,val_macro_f1,lr,"
                "grad_scale,epoch_sec,gpu_mem_gb\n")

    best_auroc, best_epoch, patience = -1.0, -1, 0
    for epoch in range(cfg.train.epochs):
        model.train()
        t0 = time.perf_counter()
        running, seen = 0.0, 0
        for x, y in loaders["train"]:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=cfg.train.amp):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            if cfg.train.grad_clip:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * len(x)
            seen += len(x)
        scheduler.step()

        train_loss = running / seen
        val_loss, val_m = evaluate(model, loaders["val"], criterion, device,
                                   cfg.train.amp)
        dt = time.perf_counter() - t0
        mem = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0
        lr_now = optimizer.param_groups[0]["lr"]
        gs = scaler.get_scale() if cfg.train.amp else 0.0

        print(f"[{epoch+1:3d}/{cfg.train.epochs}] train={train_loss:.4f} "
              f"val={val_loss:.4f} {format_metrics(val_m)} "
              f"lr={lr_now:.2e} scale={gs:.0f} {dt:.1f}s {mem:.2f}GB")
        with open(curves_path, "a") as f:
            f.write(f"{epoch+1},{train_loss:.6f},{val_loss:.6f},"
                    f"{val_m['macro_auroc']:.6f},{val_m['macro_f1']:.6f},"
                    f"{lr_now:.6e},{gs:.0f},{dt:.2f},{mem:.3f}\n")

        if val_m["macro_auroc"] > best_auroc:
            best_auroc, best_epoch, patience = val_m["macro_auroc"], epoch + 1, 0
            torch.save(model.state_dict(), os.path.join(out_dir, "best.pt"))
        else:
            patience += 1
            if patience >= cfg.train.early_stop_patience:
                print(f"[early stop] {cfg.train.early_stop_patience} epoch 개선 없음")
                break

    print(f"[best] epoch {best_epoch}, val macroAUROC {best_auroc:.4f}")

    # 테스트는 학습이 완전히 끝난 뒤 최고 체크포인트로 1회만 연다.
    model.load_state_dict(
        torch.load(os.path.join(out_dir, "best.pt"), weights_only=True))
    test_loss, test_m = evaluate(model, loaders["test"], criterion, device,
                                 cfg.train.amp)
    _, val_m = evaluate(model, loaders["val"], criterion, device, cfg.train.amp)
    print(f"[test] loss={test_loss:.4f} {format_metrics(test_m)}")

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({
            "exp_name": cfg.exp_name,
            "best_epoch": best_epoch,
            "epochs_run": epoch + 1,
            "n_params": n_params,
            "feature_shape": list(feat.shape),
            "val": val_m,
            "test": {**test_m, "loss": test_loss},
            "peak_gpu_mem_gb": round(
                torch.cuda.max_memory_allocated() / 1e9, 3)
            if device.type == "cuda" else None,
        }, f, indent=2)
    print(f"[done] {out_dir}")


if __name__ == "__main__":
    main()
