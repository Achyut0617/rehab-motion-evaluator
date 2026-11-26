"""Phase 9: Background hardware telemetry monitor using Jetson tegrastats (with Linux fallback)."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import threading
import time
from typing import Any, Dict, Optional

_RAM_RE = re.compile(r"RAM\s+(\d+)/(\d+)MB")
_CPU_RE = re.compile(r"CPU\s+\[([^\]]+)\]")
_GPU_RE = re.compile(r"GR3D_FREQ\s+(\d+)%")
_TEMP_RE = re.compile(r"(?:tj|CPU|GPU|soc0)@([\d.]+)C", re.IGNORECASE)
_POWER_RE = re.compile(r"(?:VDD_IN|POM_5V_IN)\s+(\d+)mW", re.IGNORECASE)


def parse_tegrastats_line(line: str) -> Dict[str, Any]:
    """Parse a single tegrastats output line into structured metrics."""
    stats: Dict[str, Any] = {
        "source": "tegrastats",
        "cpu_percent": None,
        "gpu_percent": None,
        "ram_used_mb": None,
        "ram_total_mb": None,
        "ram_percent": None,
        "temp_c": None,
        "power_w": None,
    }

    ram_m = _RAM_RE.search(line)
    if ram_m:
        used, total = int(ram_m.group(1)), int(ram_m.group(2))
        stats["ram_used_mb"] = used
        stats["ram_total_mb"] = total
        stats["ram_percent"] = round((used / total) * 100.0, 1) if total > 0 else 0.0

    cpu_m = _CPU_RE.search(line)
    if cpu_m:
        loads = [int(v) for v in re.findall(r"(\d+)%", cpu_m.group(1))]
        if loads:
            stats["cpu_percent"] = round(sum(loads) / len(loads), 1)

    gpu_m = _GPU_RE.search(line)
    if gpu_m:
        stats["gpu_percent"] = int(gpu_m.group(1))

    temps = [float(t) for t in _TEMP_RE.findall(line)]
    if temps:
        stats["temp_c"] = round(max(temps), 1)

    pwr_m = _POWER_RE.search(line)
    if pwr_m:
        stats["power_w"] = round(int(pwr_m.group(1)) / 1000.0, 2)

    return stats


def _read_linux_proc_stats() -> Dict[str, Any]:
    """Fallback system metrics when running on a non-Jetson Linux machine."""
    ram_used_mb: Optional[int] = None
    ram_total_mb: Optional[int] = None
    ram_percent: Optional[float] = None
    cpu_percent: Optional[float] = None

    try:
        meminfo: Dict[str, int] = {}
        with open("/proc/meminfo", "r", encoding="utf-8") as fh:
            for row in fh:
                parts = row.split()
                if len(parts) >= 2:
                    meminfo[parts[0].rstrip(":")] = int(parts[1])
        total_kb = meminfo.get("MemTotal", 0)
        avail_kb = meminfo.get("MemAvailable", meminfo.get("MemFree", 0))
        if total_kb > 0:
            used_kb = max(total_kb - avail_kb, 0)
            ram_total_mb = total_kb // 1024
            ram_used_mb = used_kb // 1024
            ram_percent = round((used_kb / total_kb) * 100.0, 1)
    except OSError:
        pass

    try:
        load1 = os.getloadavg()[0]
        cores = os.cpu_count() or 1
        cpu_percent = round(min(100.0, (load1 / cores) * 100.0), 1)
    except (OSError, AttributeError):  # getloadavg is unavailable on Windows
        pass

    return {
        "source": "linux_proc",
        "cpu_percent": cpu_percent,
        "gpu_percent": None,
        "ram_used_mb": ram_used_mb,
        "ram_total_mb": ram_total_mb,
        "ram_percent": ram_percent,
        "temp_c": None,
        "power_w": None,
    }


class JetsonHardwareMonitor:
    """Continuously streams Jetson tegrastats (or Linux fallback) in a background thread."""

    def __init__(self, interval_ms: int = 1000) -> None:
        self.interval_ms = interval_ms
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._proc: Optional[subprocess.Popen[str]] = None
        self._stats: Dict[str, Any] = {
            "source": "initializing",
            "cpu_percent": None,
            "gpu_percent": None,
            "ram_used_mb": None,
            "ram_total_mb": None,
            "ram_percent": None,
            "temp_c": None,
            "power_w": None,
        }

    def start(self) -> "JetsonHardwareMonitor":
        if self._running:
            return self
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        return self

    def _run_loop(self) -> None:
        tegrastats_bin = shutil.which("tegrastats")
        if tegrastats_bin:
            try:
                self._proc = subprocess.Popen(
                    [tegrastats_bin, "--interval", str(self.interval_ms)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                )
                while self._running and self._proc.stdout is not None:
                    line = self._proc.stdout.readline()
                    if not line:
                        break
                    parsed = parse_tegrastats_line(line)
                    with self._lock:
                        self._stats = parsed
                return
            except Exception:
                pass

        # Fallback loop on non-Jetson Linux
        while self._running:
            parsed = _read_linux_proc_stats()
            with self._lock:
                self._stats = parsed
            time.sleep(max(self.interval_ms / 1000.0, 1.0))

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._stats)

    def stop(self) -> None:
        self._running = False
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=2)
            except Exception:
                pass
            self._proc = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
