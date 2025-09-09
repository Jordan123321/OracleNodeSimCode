from contextlib import contextmanager
from pathlib import Path
import json, time, random, numpy as np

def ensure_dir(p: str | Path) -> Path:
    p = Path(p); p.mkdir(parents=True, exist_ok=True); return p

def save_json(data, path: str | Path):
    Path(path).write_text(json.dumps(data, indent=2))

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

@contextmanager
def tic(msg: str = ""):
    t0 = time.time()
    yield
    dt = time.time() - t0
    if msg:
        print(f"{msg}: {dt:.3f}s")