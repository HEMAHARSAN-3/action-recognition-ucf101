import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set all random seeds.

    Args:
        seed (int): seed value
    """

    # Python
    random.seed(seed)

    # Environment
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)

    # CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"[INFO] Seed set to {seed}")