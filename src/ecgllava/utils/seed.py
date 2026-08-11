"""시드 고정. 재현성이 이 프로젝트의 품질 지표라 결정론 옵션까지 켠다."""

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    """DataLoader worker 별 시드. worker 마다 다르되 실행 간에는 동일하게."""
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)
