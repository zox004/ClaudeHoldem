"""M1 Pro MPS availability + sanity check.

Run: `uv run python scripts/check_mps.py` (or plain `python3 scripts/check_mps.py`).
Exits non-zero if MPS is not usable, so it can gate Phase 0 exit criteria.
"""

from __future__ import annotations

import platform
import sys
import time


def main() -> int:
    try:
        import torch
    except ImportError:
        print("[FAIL] torch not installed. Run: uv add torch")
        return 1

    print(f"platform     : {platform.platform()}")
    print(f"python       : {sys.version.split()[0]}")
    print(f"torch        : {torch.__version__}")
    print(f"mps.is_built : {torch.backends.mps.is_built()}")
    print(f"mps.is_avail : {torch.backends.mps.is_available()}")

    if not torch.backends.mps.is_available():
        print("[FAIL] MPS not available on this machine.")
        return 1

    device = torch.device("mps")
    mps_mod = getattr(torch, "mps", None)
    sync = getattr(mps_mod, "synchronize", lambda: None)
    if mps_mod is None:
        print("[WARN] torch.mps module missing. Upgrade torch to >=2.4 (CLAUDE.md requirement).")

    a = torch.randn(1024, 1024, device=device)
    b = torch.randn(1024, 1024, device=device)
    sync()
    t0 = time.perf_counter()
    c = a @ b
    sync()
    dt_ms = (time.perf_counter() - t0) * 1000
    print(f"matmul 1024² : {dt_ms:.2f} ms on {device} (result sum={c.sum().item():.2f})")

    x = torch.randn(64, 16, device=device, requires_grad=True)
    w = torch.randn(16, 4, device=device)
    (x @ w).pow(2).mean().backward()
    assert x.grad is not None and x.grad.device.type == "mps"
    print("autograd     : backward on MPS OK")

    print("[OK] MPS is available and functional.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
