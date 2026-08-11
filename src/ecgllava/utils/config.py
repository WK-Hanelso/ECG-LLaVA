"""configs/exp_*.py 파일 경로를 받아 그 안의 `cfg` 심볼을 불러온다."""

import importlib.util
import os
import sys


def load_cfg(path: str):
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    spec = importlib.util.spec_from_file_location("_exp_cfg", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "cfg"):
        raise AttributeError(f"{path} 에 `cfg` 심볼이 없다")
    return module.cfg
