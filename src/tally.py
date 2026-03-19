import os
import subprocess
import threading
import time
from typing import Dict, List, Optional, Tuple


_TALLY_LIFECYCLE_LOCK = threading.Lock()
_TALLY_USERS = 0
_TALLY_STARTED_BY_HELPER = False


def _get_tally_log_paths() -> Tuple[str, str]:
    """Resolve log file locations for iox-roudi and tally server."""
    results_dir = None
    try:
        import src.globals as globals
        results_dir = globals.get_results_dir()
    except Exception:
        # Fallback for callers that haven't initialized global results dir.
        results_dir = os.path.join(os.getcwd(), "results")

    os.makedirs(results_dir, exist_ok=True)
    return (
        os.path.join(results_dir, "tally_iox.log"),
        os.path.join(results_dir, "tally_server.log"),
    )


def detect_tally_root(repo_dir: str) -> str:
    env_tally = os.environ.get("TALLY_ROOT") or os.environ.get("TALLY_HOME")
    if env_tally and os.path.isdir(env_tally):
        return env_tally

    repo_tally = os.path.abspath(os.path.join(repo_dir, "..", "tally"))
    if os.path.isdir(repo_tally):
        return repo_tally

    fallback = "/home/tally-bench/tally"
    if os.path.isdir(fallback):
        return fallback

    raise FileNotFoundError("Cannot locate tally root. Set TALLY_ROOT or TALLY_HOME.")


def is_tally_server_running(tally_root: str) -> bool:
    query_script = os.path.join(tally_root, "scripts", "query_server.sh")
    if not os.path.exists(query_script):
        return False
    rc = subprocess.run(["bash", query_script], capture_output=True, text=True).returncode
    return rc == 0


def is_iox_running() -> bool:
    return subprocess.run(["pgrep", "-f", "iox-roudi|roudi"], capture_output=True, text=True).returncode == 0


def ensure_tally_runtime(scheduler: Optional[str], repo_dir: str) -> Optional[str]:
    global _TALLY_USERS
    global _TALLY_STARTED_BY_HELPER

    if scheduler not in {"tally", "tgs", "naive"}:
        return None

    tally_root = detect_tally_root(repo_dir)
    os.environ["TALLY_HOME"] = tally_root
    print(f"[tally] using tally root: {tally_root}")

    with _TALLY_LIFECYCLE_LOCK:
        if _TALLY_USERS > 0:
            _TALLY_USERS += 1
            print(f"[tally] reusing existing tally runtime (users={_TALLY_USERS})")
            return tally_root

        started_by_this_flow = False

        start_iox_script = os.path.join(tally_root, "scripts", "start_iox.sh")
        start_server_script = os.path.join(tally_root, "scripts", "start_server.sh")
        if not os.path.exists(start_iox_script) or not os.path.exists(start_server_script):
            raise FileNotFoundError("Missing tally start scripts under tally/scripts")

        if not is_iox_running():
            print("[tally] starting iox-roudi ...")
            iox_log_path, _ = _get_tally_log_paths()
            print(f"[tally] iox logs: {iox_log_path}")
            with open(iox_log_path, "a") as iox_log:
                subprocess.Popen(
                    ["bash", start_iox_script],
                    stdout=iox_log,
                    stderr=iox_log,
                    start_new_session=True,
                )
            time.sleep(15)
            started_by_this_flow = True
            print("[tally] iox-roudi start requested")
        else:
            print("[tally] iox-roudi already running")

        if not is_tally_server_running(tally_root):
            if scheduler == "tally":
                scheduler_policy = "PRIORITY"
            elif scheduler == "tgs":
                scheduler_policy = "TGS"
            else:
                scheduler_policy = "NAIVE"
            print(f"[tally] starting tally server with SCHEDULER_POLICY={scheduler_policy} ...")
            server_env = os.environ.copy()
            server_env["SCHEDULER_POLICY"] = scheduler_policy
            _, server_log_path = _get_tally_log_paths()
            print(f"[tally] server logs: {server_log_path}")
            with open(server_log_path, "a") as server_log:
                subprocess.Popen(
                    ["bash", start_server_script],
                    stdout=server_log,
                    stderr=server_log,
                    start_new_session=True,
                    env=server_env,
                )
            time.sleep(2)
            started_by_this_flow = True
            print("[tally] tally server start requested")
        else:
            print("[tally] tally server already running")

        if not is_tally_server_running(tally_root):
            raise RuntimeError("Failed to start tally server runtime.")

        _TALLY_USERS = 1
        _TALLY_STARTED_BY_HELPER = started_by_this_flow
        print("[tally] tally runtime ready")
        return tally_root


def release_tally_runtime(scheduler: Optional[str], repo_dir: str):
    global _TALLY_USERS
    global _TALLY_STARTED_BY_HELPER

    if scheduler not in {"tally", "tgs", "naive"}:
        return

    with _TALLY_LIFECYCLE_LOCK:
        if _TALLY_USERS > 0:
            _TALLY_USERS -= 1
        print(f"[tally] release requested (remaining_users={_TALLY_USERS})")

        if _TALLY_USERS == 0 and _TALLY_STARTED_BY_HELPER:
            tally_root = detect_tally_root(repo_dir)
            print("[tally] shutting down tally server and iox-roudi started by helper")
            subprocess.run(["bash", os.path.join(tally_root, "scripts", "kill_server.sh")],
                           capture_output=True, text=True)
            subprocess.run(["bash", os.path.join(tally_root, "scripts", "kill_iox.sh")],
                           capture_output=True, text=True)
            _TALLY_STARTED_BY_HELPER = False
            print("[tally] tally runtime shutdown complete")


def wrap_command_with_tally_client(command: List[str], tally_root: str, priority: int, env: Optional[Dict[str, str]] = None) -> Tuple[List[str], Dict[str, str]]:
    wrapped_env = dict(env) if env is not None else os.environ.copy()
    wrapped_env["PRIORITY"] = str(priority)
    start_client_script = os.path.join(tally_root, "scripts", "start_client.sh")
    wrapped_cmd = ["env", f"PRIORITY={priority}", "bash", start_client_script] + command
    print(f"[tally] wrapping command with tally client: {' '.join(wrapped_cmd)}")
    return wrapped_cmd, wrapped_env