"""Memory safeguards: refuse to start, or stop, analysis when the machine is short of memory.

On macOS the kernel's memory-pressure level is read with ``sysctl`` (1 normal, 2 warning, 4 critical);
elsewhere only the process's own peak resident memory is checked.

Two policies:

* ``standard`` (default): work continues at the warning level, which macOS reports routinely on 8 GB
  machines with a browser and an editor open, and the run records that it happened; work stops at the
  critical level, where the kernel starts terminating processes.
* ``strict``: anything above normal stops work. Used for benchmarks, so measurements are never taken on
  a machine that is already swapping.

The process memory budget applies under both.
"""
from __future__ import annotations

import os
import resource
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional

NORMAL, WARNING, CRITICAL = 1, 2, 4


class ResourceExhausted(RuntimeError):
    pass


def memory_pressure_level() -> Optional[int]:
    if sys.platform != "darwin":
        return None
    try:
        out = subprocess.run(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
                             capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return int(out) if out.isdigit() else None


def peak_rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2**20 if sys.platform == "darwin" else r / 1024


def physical_memory_mb() -> Optional[float]:
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 2**20
    except (ValueError, OSError, AttributeError):
        return None


@dataclass
class ResourceGuard:
    """Checks memory before and periodically during a run.

    The RSS budget defaults to half of physical memory, leaving room for the OS, browser and app on
    8 GB machines.
    """
    policy: Literal["standard", "strict"] = "standard"
    rss_budget_mb: Optional[float] = None
    check_every_s: float = 2.0
    pressure: Callable[[], Optional[int]] = memory_pressure_level
    rss: Callable[[], float] = peak_rss_mb
    warnings: list[str] = field(default_factory=list)
    _last_check: float = float("-inf")

    def __post_init__(self) -> None:
        if self.policy not in ("standard", "strict"):
            raise ValueError(f"unknown resource policy {self.policy!r}")
        if self.rss_budget_mb is None:
            total = physical_memory_mb()
            self.rss_budget_mb = total * 0.5 if total else 4096.0

    @property
    def stop_above(self) -> int:
        return NORMAL if self.policy == "strict" else WARNING

    def check(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_check < self.check_every_s:
            return
        self._last_check = now
        level = self.pressure()
        if level is not None and level > self.stop_above:
            raise ResourceExhausted(f"system memory pressure level {level} ({self.policy} policy stops above "
                                    f"{self.stop_above})")
        if level is not None and level > NORMAL and not self.warnings:
            self.warnings.append(f"system memory pressure reached level {level}; analysis continued")
        rss = self.rss()
        if rss > self.rss_budget_mb:
            raise ResourceExhausted(f"peak memory {rss:.0f} MB exceeds budget {self.rss_budget_mb:.0f} MB")
