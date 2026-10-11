from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from anelench.learned import train

if __name__ == "__main__":
    threads = int(sys.argv[1]) if len(sys.argv) > 1 else torch.get_num_threads()
    torch.set_num_threads(threads)
    for name, p in (("rings", 0.0), ("knots", 0.5)):
        t0 = time.time()
        out = train(name, p, seed=0 if name == "rings" else 1, log=lambda m: print(m, flush=True))
        print(f"saved {out} ({time.time() - t0:.0f}s)", flush=True)
