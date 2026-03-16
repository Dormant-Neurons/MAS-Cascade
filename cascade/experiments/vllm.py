"""
Utilities for discovering and launching local vLLM servers.
"""

from __future__ import annotations

import os
import shlex
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from urllib.error import URLError
from urllib.request import urlopen


@dataclass
class VLLMServerInfo:
    gpu_index: int
    pid: int
    port: Optional[int]
    command: str

    @property
    def base_url(self) -> Optional[str]:
        if self.port is None:
            return None
        return f"http://localhost:{self.port}/v1"


def _run_command(cmd: List[str]) -> str:
    return subprocess.check_output(cmd, text=True).strip()


def _parse_gpu_listing(output: str) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        index_str, uuid = parts[0], parts[1]
        try:
            mapping[uuid] = int(index_str)
        except ValueError:
            continue
    return mapping


def _parse_port_from_command(command: str) -> Optional[int]:
    try:
        tokens = shlex.split(command)
    except ValueError:
        return None
    for idx, token in enumerate(tokens):
        if token == "--port" and idx + 1 < len(tokens):
            try:
                return int(tokens[idx + 1])
            except ValueError:
                return None
        if token.startswith("--port="):
            try:
                return int(token.split("=", 1)[1])
            except ValueError:
                return None
    return None


def _gpu_process_map() -> Dict[int, List[tuple[int, str]]]:
    """Return mapping from GPU index to list of (pid, process_name)."""
    try:
        gpu_listing = _run_command(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"]
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {}

    uuid_to_index = _parse_gpu_listing(gpu_listing)
    process_map: Dict[int, List[tuple[int, str]]] = {
        idx: [] for idx in uuid_to_index.values()
    }

    try:
        apps_output = _run_command(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name",
                "--format=csv,noheader",
            ]
        )
    except subprocess.CalledProcessError:
        return process_map

    for line in apps_output.splitlines():
        if not line.strip() or "No running" in line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        uuid, pid_str, proc_name = parts[:3]
        idx = uuid_to_index.get(uuid)
        if idx is None:
            continue
        try:
            pid = int(pid_str)
        except ValueError:
            continue
        process_map.setdefault(idx, []).append((pid, proc_name))
    return process_map


def list_vllm_servers() -> List[VLLMServerInfo]:
    """Return currently running vLLM servers detected on the system."""
    try:
        gpu_map = _parse_gpu_listing(
            _run_command(
                [
                    "nvidia-smi",
                    "--query-gpu=index,uuid",
                    "--format=csv,noheader",
                ]
            )
        )
    except FileNotFoundError:
        return []
    except subprocess.CalledProcessError:
        return []

    try:
        apps_output = _run_command(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name",
                "--format=csv,noheader",
            ]
        )
    except subprocess.CalledProcessError:
        apps_output = ""

    servers: List[VLLMServerInfo] = []
    for line in apps_output.splitlines():
        if not line.strip() or "No running" in line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        uuid, pid_str, proc_name = parts[:3]
        idx = gpu_map.get(uuid)
        if idx is None:
            continue
        try:
            pid = int(pid_str)
        except ValueError:
            continue
        try:
            cmd = _run_command(["ps", "-p", str(pid), "-o", "command="])
        except subprocess.CalledProcessError:
            continue
        if "vllm.entrypoints.openai.api_server" not in cmd:
            continue
        servers.append(
            VLLMServerInfo(
                gpu_index=idx,
                pid=pid,
                port=_parse_port_from_command(cmd),
                command=cmd,
            )
        )
    return servers


def list_gpus() -> List[int]:
    """Return indices of available GPUs."""
    try:
        output = _run_command(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"]
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    indices: List[int] = []
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            indices.append(int(line))
        except ValueError:
            continue
    return indices


def find_available_gpu(exclude: Iterable[int] = ()) -> Optional[int]:
    """Return the first GPU index without a running vLLM server."""
    exclude_set = set(exclude)
    for idx in list_gpus():
        if idx not in exclude_set:
            return idx
    return None


def find_free_port(start: int = 8000, limit: int = 100) -> Optional[int]:
    """Find an available localhost port."""
    for port in range(start, start + limit):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    return None


def wait_for_ready(port: int, timeout: int = 120, interval: int = 5) -> bool:
    """Poll the vLLM server until it responds or timeout occurs."""
    url = f"http://127.0.0.1:{port}/v1/models"
    deadline = time.time() + max(timeout, 1)
    while time.time() < deadline:
        try:
            with urlopen(url, timeout=interval) as resp:
                if resp.status == 200:
                    return True
        except URLError:
            pass
        except Exception:
            pass
        time.sleep(interval)
    return False


def ensure_vllm_server(
    model: str,
    *,
    requested_gpu: Optional[int] = None,
    explicit_port: Optional[int] = None,
    port_start: int = 8000,
    log_dir: str = "logs/vllm",
    gpu_memory_utilization: Optional[float] = None,
    max_model_len: Optional[int] = None,
    extra_args: Optional[str] = None,
    wait_seconds: int = 120,
) -> VLLMServerInfo:
    """
    Ensure a vLLM server is running and return its information.

    If an existing server already matches the requested GPU/port, it is reused.
    Otherwise a new server is launched.
    """
    servers = list_vllm_servers()
    used_gpus = {server.gpu_index for server in servers}
    used_ports = {server.port for server in servers if server.port is not None}
    process_map = _gpu_process_map()

    # Reuse existing server if it matches user preference
    if explicit_port is not None:
        for server in servers:
            if server.port == explicit_port:
                return server
    if requested_gpu is not None:
        for server in servers:
            if server.gpu_index == requested_gpu:
                return server

        active = process_map.get(requested_gpu, [])
        if active:
            details = ", ".join(f"{pid}:{name}" for pid, name in active)
            raise SystemExit(
                f"ERROR: GPU {requested_gpu} already has running processes ({details}). "
                "Stop them or choose a different GPU for vLLM."
            )

    gpu_index = requested_gpu
    if gpu_index is None:
        for candidate in list_gpus():
            if candidate in used_gpus:
                continue
            if process_map.get(candidate):
                continue
            gpu_index = candidate
            break
    if gpu_index is None:
        busy_info = []
        for idx in sorted(process_map.keys()):
            procs = process_map.get(idx) or []
            if not procs:
                continue
            details = ", ".join(f"{pid}:{name}" for pid, name in procs)
            busy_info.append(f"{idx} ({details})")
        suffix = (
            f" Busy GPUs: {', '.join(busy_info)}."
            if busy_info
            else ""
        )
        raise SystemExit(
            "ERROR: No free GPU found to launch vLLM. "
            "Either stop existing servers or specify --vllm-gpu to reuse one."
            + suffix
        )

    if explicit_port is not None and explicit_port in used_ports:
        raise SystemExit(f"ERROR: Requested port {explicit_port} is already in use by another vLLM server.")

    port = explicit_port
    if port is None:
        candidate = find_free_port(start=port_start)
        if candidate is None:
            raise SystemExit("ERROR: Unable to find a free port for vLLM server.")
        port = candidate

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"vllm_gpu{gpu_index}_port{port}_{timestamp}.log"

    cmd = [
        "python",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model,
        "--port",
        str(port),
    ]
    if gpu_memory_utilization is not None:
        cmd.extend(["--gpu-memory-utilization", str(gpu_memory_utilization)])
    if max_model_len is not None:
        cmd.extend(["--max-model-len", str(max_model_len)])
    if extra_args:
        cmd.extend(shlex.split(extra_args))

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)

    with open(log_file, "w", encoding="utf-8") as fh:
        process = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)

    if not wait_for_ready(port, timeout=wait_seconds, interval=5):
        process.terminate()
        raise SystemExit(
            f"ERROR: vLLM server on GPU {gpu_index} failed to become ready. "
            f"Check log file: {log_file}"
        )

    return VLLMServerInfo(
        gpu_index=gpu_index,
        pid=process.pid,
        port=port,
        command=" ".join(cmd),
    )
