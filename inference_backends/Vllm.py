import sys
import os
import subprocess
import threading

repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_dir)

import src.utils as utils
import src.globals as globals

class Vllm:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(Vllm, cls).__new__(cls)
                    cls.__initialized = False
        return cls._instance

    def __init__(self):
        if self.__initialized:
            return

        self.refcount = 0
        self.lock = threading.Lock()
        self.__initialized = True

    def launch_backend(self, *args, **kwargs):
        print("Launching vLLM backend")
        api_port = kwargs.get('api_port', 8080)
        model = kwargs.get('model', 'meta-llama/Llama-3.2-3B-Instruct')
        device = kwargs.get('device', 'gpu')
        vllm_path = kwargs.get('vllm_path', f"{repo_dir}/inference_backends/vllm")

        if not vllm_path.startswith("/"):
            vllm_path = os.path.join(repo_dir, vllm_path)

        with self.lock:
            self.refcount += 1
            if self.refcount > 1:
                print("vLLM backend already running")
                return {"status": "backend_already_running"}

            utils.util_run_server_script_check_log(
                script_path=f"{repo_dir}/inference_backends/vllm_server.sh",
                server_dir=vllm_path,
                stdout_log_path=f"{globals.get_results_dir()}/vllm_server_stdout",
                stderr_log_path=f"{globals.get_results_dir()}/vllm_server_stderr",
                stderr_ready_patterns=[],
                stdout_ready_patterns=["Application startup complete"],
                listen_port=api_port,
                api_port=api_port,
                model=model,
                device=device,
                mps=100
            )

            print("vLLM backend launched")
            return {"status": "backend_launched"}

    def cleanup_backend(self, *args, **kwargs):
        print("Cleaning up vLLM backend")
        api_port = kwargs.get('api_port', 8080)
        with self.lock:
            self.refcount -= 1
            if self.refcount == 0:
                print("Cleaning up vLLM backend")
                process = subprocess.Popen(
                    [f"{repo_dir}/scripts/cleanup.sh", str(api_port)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                process.wait()
                return {"status": "backend_cleaned_up"}
            else:
                print("vLLM backend still running")
                return {"status": "backend_still_running"}
