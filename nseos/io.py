"""Small I/O helpers: JSON with numpy support, pickles, timings, cached stages."""
import json
import os
import pickle
import time

import numpy as np


def to_jsonable(o):
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return to_jsonable(o.tolist())
    if isinstance(o, (np.floating,)):
        v = float(o)
        return None if not np.isfinite(v) else v
    if isinstance(o, float):
        return None if not np.isfinite(o) else o
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if hasattr(o, "detach"):  # torch tensor
        return to_jsonable(o.detach().cpu().numpy())
    return o


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(to_jsonable(obj), fh, indent=2)
    os.replace(tmp, path)
    return path


def load_json(path, default=None):
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return default


def save_pickle(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def load_pickle(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


class Timings:
    """Persistent wall-clock ledger used by the amortisation analysis."""

    def __init__(self, path):
        self.path = path
        self.data = load_json(path, default={}) or {}

    def set(self, key, seconds):
        self.data[key] = float(seconds)
        save_json(self.path, self.data)

    def get(self, key, default=None):
        return self.data.get(key, default)


class timer:
    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.time() - self.t0
        return False


def cached(path, force=False):
    """True if the stage output already exists and should be reused."""
    if os.path.exists(path) and not force:
        print(f"[cache] reusing {path}")
        return True
    return False
