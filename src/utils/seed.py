import os, random, numpy as np
try:
    import tensorflow as tf
except Exception:  # pragma: no cover
    tf = None

def set_global_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if tf is not None:
        tf.random.set_seed(seed)
