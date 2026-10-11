from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/eventhorizontelescope/2019-D01-01/master/uvfits/"
FILES = {
    "SR1_M87_2017_095_hi_hops_netcal_StokesI.uvfits": 567360,
    "SR1_M87_2017_095_lo_hops_netcal_StokesI.uvfits": 567360,
    "SR1_M87_2017_096_hi_hops_netcal_StokesI.uvfits": 786240,
    "SR1_M87_2017_096_lo_hops_netcal_StokesI.uvfits": 751680,
    "SR1_M87_2017_100_hi_hops_netcal_StokesI.uvfits": 244800,
    "SR1_M87_2017_100_lo_hops_netcal_StokesI.uvfits": 224640,
    "SR1_M87_2017_101_hi_hops_netcal_StokesI.uvfits": 702720,
    "SR1_M87_2017_101_lo_hops_netcal_StokesI.uvfits": 650880,
}
DEST = Path(__file__).resolve().parents[1] / "data" / "eht2017"


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    for name, size in FILES.items():
        out = DEST / name
        if out.exists() and out.stat().st_size == size:
            print(f"ok      {name}")
            continue
        print(f"fetch   {name} ({size / 1024:.0f} KB)")
        with urllib.request.urlopen(BASE + name, timeout=60) as resp:
            payload = resp.read()
        if len(payload) != size:
            print(f"size mismatch for {name}: got {len(payload)}, expected {size}", file=sys.stderr)
            return 1
        out.write_bytes(payload)
    print(f"EHT 2017 M87 data ready in {DEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
