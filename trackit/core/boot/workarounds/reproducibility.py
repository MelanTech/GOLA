def seed_all_rng(seed=0):
    import random
    random.seed(seed)
    import numpy as np
    np.random.seed(random.randint(0, 255))
    import torch
    torch.manual_seed(random.randint(0, 255))

    # Zekai Shao: set the seed for all GPUs
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random.randint(0, 255))


def enable_deterministic_computation():
    import torch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Zekai Shao: enable deterministic computation
    torch.use_deterministic_algorithms(True)
