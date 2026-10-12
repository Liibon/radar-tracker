#!/usr/bin/env python3
"""Record CUDA environment evidence without treating fallback as GPU support.

This probe only establishes whether an NVIDIA device is visible and whether the
CUDA compiler is installed. It does not prove that the tracker invokes CUDA.
"""

from __future__ import annotations

import json
import platform
import shutil
import subprocess


def _run(command: list[str]) -> dict[str, object]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "error": str(exc)}
    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def main() -> int:
    nvidia_smi = shutil.which("nvidia-smi")
    nvcc = shutil.which("nvcc")
    device_probe = (
        _run(
            [
                nvidia_smi,
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        )
        if nvidia_smi
        else {"ok": False, "error": "nvidia-smi not found"}
    )
    compiler_probe = _run([nvcc, "--version"]) if nvcc else {
        "ok": False,
        "error": "nvcc not found",
    }
    device_available = bool(device_probe.get("ok"))
    compiler_available = bool(compiler_probe.get("ok"))
    result = {
        "host": platform.node(),
        "platform": platform.platform(),
        "cuda_device_available": device_available,
        "nvcc_available": compiler_available,
        "cuda_build_and_run_candidate": device_available and compiler_available,
        "device_probe": device_probe,
        "compiler_probe": compiler_probe,
        "warning": (
            "This environment probe does not prove that the tracker executed a "
            "CUDA code path; runtime backend logging and output validation are required."
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
