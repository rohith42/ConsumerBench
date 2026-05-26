import os
import sys
import time
import subprocess
import threading

repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_dir)

import src.globals as globals
TGS_PATH = os.getenv('TGS_PATH')


class TGSLlamaCpp:
    _instance = None
    _class_lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._class_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls.__initialized = False
        return cls._instance

    def __init__(self):
        if self.__initialized:
            return
        self._containers = {}
        self._lock = threading.Lock()
        self.__initialized = True

    def launch_backend(self, *args, **kwargs):
        api_port = int(kwargs.get('api_port', 8080))
        priority = kwargs.get('priority', 'high')
        assert priority in ('high', 'low'), f"priority must be high|low, got {priority}"
        tgs_path = kwargs.get('tgs_path', TGS_PATH)
        model = kwargs.get('model')

        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '0').strip()
        if not cuda_visible:
            raise RuntimeError("CUDA_VISIBLE_DEVICES is set but empty; cannot pick a GPU for TGS")
        print(f"CUDA_VISIBLE_DEVICES={cuda_visible} for TGS backend")
        gpu_id = cuda_visible.split(',')[0].strip()

        ngl = kwargs.get('ngl', 99)
        parallel = kwargs.get('parallel', 4)
        ctx_size = kwargs.get('ctx_size', 131072)

        key = (priority, api_port)
        with self._lock:
            if key in self._containers:
                self._containers[key]['refcount'] += 1
                print(f"TGSLlamaCpp backend already running for {key}")
                return {"status": "backend_already_running"}

            container_name = f"llamacpp_{priority}_{api_port}"
            log_dir = os.path.join(globals.get_results_dir(), "server_logs")
            os.makedirs(log_dir, exist_ok=True)
            stdout_log = os.path.join(log_dir, f"tgs_llamacpp_stdout_{priority}_{api_port}.log")
            stderr_log = os.path.join(log_dir, f"tgs_llamacpp_stderr_{priority}_{api_port}.log")

            cmd = [
                f"{tgs_path}/scripts/launch_llamacpp_server.sh",
                priority, str(api_port),
                "--gpu", str(gpu_id),
                "--ngl", str(ngl),
                "--parallel", str(parallel),
                "--ctx", str(ctx_size),
                "--name", container_name,
            ]
            if model:
                cmd += ["--model", model]

            print(f"Launching TGSLlamaCpp ({priority}) on port {api_port}, gpu {gpu_id}, container {container_name}")
            stdout_file = open(stdout_log, 'w')
            stderr_file = open(stderr_log, 'w')
            process = subprocess.Popen(cmd, stdout=stdout_file, stderr=stderr_file, start_new_session=True)

            if not self._wait_for_ready(stderr_log, stdout_log, timeout=180):
                subprocess.run(['docker', 'stop', container_name], check=False)
                stdout_file.close()
                stderr_file.close()
                raise RuntimeError(f"TGS llama-server {container_name} failed to become ready (see {stderr_log})")

            self._containers[key] = {
                'container_name': container_name,
                'refcount': 1,
                'process': process,
                'stdout_file': stdout_file,
                'stderr_file': stderr_file,
            }
            print(f"TGSLlamaCpp backend launched ({priority}@{api_port})")
            return {"status": "backend_launched"}

    def cleanup_backend(self, *args, **kwargs):
        api_port = int(kwargs.get('api_port', 8080))
        priority = kwargs.get('priority', 'high')
        key = (priority, api_port)
        with self._lock:
            if key not in self._containers:
                print(f"TGSLlamaCpp backend not running for {key}")
                return {"status": "backend_not_running"}
            self._containers[key]['refcount'] -= 1
            if self._containers[key]['refcount'] > 0:
                print(f"TGSLlamaCpp backend still in use for {key}")
                return {"status": "backend_still_running"}

            entry = self._containers.pop(key)
            container_name = entry['container_name']
            print(f"Stopping TGSLlamaCpp container {container_name}")
            subprocess.run(['docker', 'stop', container_name], check=False)
            try:
                entry['stdout_file'].close()
                entry['stderr_file'].close()
            except Exception:
                pass
            return {"status": "backend_cleaned_up"}

    @staticmethod
    def _wait_for_ready(stderr_log, stdout_log, timeout):
        pattern = "update_slots: all slots are idle"
        start = time.time()
        while time.time() - start < timeout:
            for log in (stderr_log, stdout_log):
                try:
                    with open(log, 'r') as f:
                        if pattern in f.read():
                            return True
                except FileNotFoundError:
                    pass
            time.sleep(1)
        return False
