"""Hardware probe and resource-guarded inference/decoding benchmarks.

Each benchmark runs in its own spawned child process, one at a time, with a
wall-clock timeout and a peak-memory budget, so a misbehaving model cannot
take down the caller or overlap with another model.

Usage:
    python -m evaluation.hardware probe
    python -m evaluation.hardware onnx MODEL.onnx --providers cpu coreml --size 518 518 --iters 10
    python -m evaluation.hardware decode VIDEO --widths 1280 960 640 --stride 1 2 3 --max-frames 150
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import platform
import queue
import resource
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

from spectra.engine.resources import memory_pressure_level

PROVIDERS = {"cpu": "CPUExecutionProvider", "coreml": "CoreMLExecutionProvider", "cuda": "CUDAExecutionProvider"}


def _sysctl(key: str) -> str | None:
    try:
        return subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, timeout=5).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def probe() -> dict:
    info: dict = {
        "machine": platform.machine(), "system": platform.system(), "os_release": platform.mac_ver()[0] or platform.release(),
        "python": sys.version.split()[0], "logical_cpus": os.cpu_count(),
        "disk_free_gb": round(shutil.disk_usage(Path.cwd()).free / 1e9, 1),
    }
    if info["system"] == "Darwin":
        info.update(cpu=_sysctl("machdep.cpu.brand_string"),
                    memory_gb=round(int(_sysctl("hw.memsize") or 0) / 2**30, 1),
                    performance_cores=_sysctl("hw.perflevel0.physicalcpu"),
                    efficiency_cores=_sysctl("hw.perflevel1.physicalcpu"))
    else:
        try:
            pages, size = os.sysconf("SC_PHYS_PAGES"), os.sysconf("SC_PAGE_SIZE")
            info["memory_gb"] = round(pages * size / 2**30, 1)
        except (ValueError, OSError):
            info["memory_gb"] = None
        info["cpu"] = platform.processor() or None
    try:
        import onnxruntime as ort
        info["onnxruntime"] = {"version": ort.__version__, "providers": ort.get_available_providers()}
    except ImportError:
        info["onnxruntime"] = None
    try:
        import torch
        info["torch"] = {"version": torch.__version__, "mps": torch.backends.mps.is_available(),
                         "cuda": torch.cuda.is_available(), "threads": torch.get_num_threads()}
    except ImportError:
        info["torch"] = None
    try:
        import cv2
        info["opencv"] = {"version": cv2.__version__, "threads": cv2.getNumThreads()}
    except ImportError:
        info["opencv"] = None
    info["memory_pressure_level"] = memory_pressure_level()
    info["swap_used_mb"] = swap_used_mb()
    return info



def swap_used_mb() -> float | None:
    v = _sysctl("vm.swapusage") if sys.platform == "darwin" else None
    try:
        return float(v.split("used = ")[1].split("M")[0]) if v else None
    except (IndexError, ValueError):
        return None


def _check_pressure(max_level: int) -> None:
    level = memory_pressure_level()
    if level is not None and level > max_level:
        raise MemoryError(f"system memory pressure level {level} exceeds {max_level}")


def default_memory_budget_mb() -> float:
    """Half of physical memory: leaves room for the OS, browser and app on 8 GB machines."""
    mem = probe().get("memory_gb") or 8
    return mem * 1024 * 0.5


def _peak_rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2**20 if sys.platform == "darwin" else r / 1024


def _cpu_s() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def _onnx_child(q, model, provider, size, iters, threads, budget_mb):
    try:
        import numpy as np
        import onnxruntime as ort
        base_rss = _peak_rss_mb()
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        t0 = time.perf_counter()
        sess = ort.InferenceSession(model, so, providers=[provider, "CPUExecutionProvider"] if provider != "CPUExecutionProvider" else [provider])
        load_s = time.perf_counter() - t0
        inp = sess.get_inputs()[0]
        shape = [d if isinstance(d, int) else 1 for d in inp.shape]
        if size and len(shape) == 4:
            shape[2], shape[3] = size
        x = np.random.default_rng(0).random(shape, dtype=np.float32)
        t0 = time.perf_counter()
        sess.run(None, {inp.name: x})
        first_s = time.perf_counter() - t0
        times, cpu0, wall0 = [], _cpu_s(), time.perf_counter()
        for _ in range(iters):
            t0 = time.perf_counter()
            sess.run(None, {inp.name: x})
            times.append(time.perf_counter() - t0)
            if _peak_rss_mb() > budget_mb:
                raise MemoryError(f"peak RSS exceeded budget {budget_mb:.0f} MB")
            _check_pressure(1)
        wall = time.perf_counter() - wall0
        q.put({"ok": True, "provider_requested": provider, "providers_active": sess.get_providers(),
               "input_shape": shape, "load_s": round(load_s, 3), "first_inference_s": round(first_s, 3),
               "warm_median_ms": round(statistics.median(times) * 1000, 1),
               "warm_p90_ms": round(sorted(times)[int(0.9 * (len(times) - 1))] * 1000, 1),
               "iters": iters, "cpu_utilisation_cores": round((_cpu_s() - cpu0) / wall, 2),
               "peak_rss_mb": round(_peak_rss_mb(), 0), "baseline_rss_mb": round(base_rss, 0)})
    except Exception as exc:  # reported, not raised: the parent records the failure
        q.put({"ok": False, "provider_requested": provider, "error": f"{type(exc).__name__}: {exc}"})


def _decode_child(q, video, width, stride, max_frames):
    try:
        import cv2
        t0 = time.perf_counter()
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            raise OSError("cannot open video")
        open_s = time.perf_counter() - t0
        src_w, src_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n_read = n_kept = 0
        cpu0, t0 = _cpu_s(), time.perf_counter()
        while n_read < max_frames:
            if not cap.grab():
                break
            n_read += 1
            if (n_read - 1) % stride:
                continue
            ok, frame = cap.retrieve()
            if not ok:
                break
            if width and width < src_w:
                frame = cv2.resize(frame, (width, round(src_h * width / src_w)), interpolation=cv2.INTER_AREA)
            n_kept += 1
        wall = time.perf_counter() - t0
        cap.release()
        q.put({"ok": True, "source": [src_w, src_h], "width": width, "stride": stride, "frames_read": n_read,
               "frames_kept": n_kept, "open_s": round(open_s, 3), "decode_fps": round(n_read / wall, 1),
               "cpu_utilisation_cores": round((_cpu_s() - cpu0) / wall, 2), "peak_rss_mb": round(_peak_rss_mb(), 0)})
    except Exception as exc:
        q.put({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


def run_isolated(target, args, timeout_s: float) -> dict:
    """Run one benchmark in a spawned child; terminate it on timeout."""
    level = memory_pressure_level()
    if level is not None and level > 1:
        return {"ok": False, "error": f"not started: system memory pressure level {level}", "wall_s": 0.0}
    swap_before = swap_used_mb()
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=target, args=(q, *args), daemon=True)
    started = time.perf_counter()
    p.start()
    deadline = started + timeout_s
    result = None
    try:
        while result is None:
            try:
                result = q.get(timeout=0.2)
            except queue.Empty:
                if not p.is_alive() and q.empty():
                    result = {"ok": False, "error": f"child exited with code {p.exitcode} without a result"}
                elif time.perf_counter() > deadline:
                    result = {"ok": False, "error": f"timeout after {timeout_s}s"}
    finally:
        p.join(5)
        if p.is_alive():
            p.terminate()
            p.join(5)
    result["wall_s"] = round(time.perf_counter() - started, 2)
    swap_after = swap_used_mb()
    if swap_before is not None and swap_after is not None:
        result["swap_delta_mb"] = round(swap_after - swap_before, 1)
    result["memory_pressure_after"] = memory_pressure_level()
    return result


def bench_onnx(model: Path, providers: list[str], size=None, iters: int = 10, threads: int = 0,
               timeout_s: float = 300, budget_mb: float | None = None) -> list[dict]:
    model = Path(model)
    budget_mb = budget_mb or default_memory_budget_mb()
    size_mb = model.stat().st_size / 2**20
    if size_mb * 3 > budget_mb:
        return [{"ok": False, "error": f"model {size_mb:.0f} MB likely exceeds budget {budget_mb:.0f} MB; skipped"}]
    out = []
    for name in providers:  # strictly sequential
        r = run_isolated(_onnx_child, (str(model), PROVIDERS[name], size, iters, threads, budget_mb), timeout_s)
        r.update(model=model.name, model_mb=round(size_mb, 1), threads=threads or "default")
        out.append(r)
    return out


def bench_decode(video: Path, widths: list[int], strides: list[int], max_frames: int = 150,
                 timeout_s: float = 120) -> list[dict]:
    return [run_isolated(_decode_child, (str(video), w, s, max_frames), timeout_s) for w in widths for s in strides]


def main() -> None:
    p = argparse.ArgumentParser(description="Spectra hardware benchmark")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("probe")
    o = sub.add_parser("onnx")
    o.add_argument("model", type=Path)
    o.add_argument("--providers", nargs="+", default=["cpu"], choices=sorted(PROVIDERS))
    o.add_argument("--size", nargs=2, type=int)
    o.add_argument("--iters", type=int, default=10)
    o.add_argument("--threads", type=int, default=0)
    o.add_argument("--timeout", type=float, default=300)
    d = sub.add_parser("decode")
    d.add_argument("video", type=Path)
    d.add_argument("--widths", nargs="+", type=int, default=[0])
    d.add_argument("--stride", nargs="+", type=int, default=[1])
    d.add_argument("--max-frames", type=int, default=150)
    a = p.parse_args()
    if a.cmd == "probe":
        res = probe()
    elif a.cmd == "onnx":
        res = {"hardware": probe(), "results": bench_onnx(a.model, a.providers, a.size, a.iters, a.threads, a.timeout)}
    else:
        res = {"hardware": probe(), "results": bench_decode(a.video, a.widths, a.stride, a.max_frames)}
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
