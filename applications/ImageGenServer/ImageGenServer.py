import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from typing import Any, Dict

from datasets import load_dataset
import src.globals as globals

repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(repo_dir)

from applications.application import Application


class ImageGenServer(Application):
    def __init__(self):
        super().__init__()
        self.imagegen_prompts = []
        self.server_process = None
        self.server_stdout = None
        self.server_stderr = None
        self._setup_scheduler = None
        self.api_port = None

    def _stop_server_process(self):
        if not self.server_process:
            return

        try:
            if self.server_process.poll() is None:
                # Kill the entire process group because tally wrapping may add
                # intermediate shell processes.
                os.killpg(self.server_process.pid, signal.SIGTERM)
                self.server_process.wait(timeout=10)
        except ProcessLookupError:
            pass
        except subprocess.TimeoutExpired:
            try:
                os.killpg(self.server_process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        finally:
            self.server_process = None

    def _read_log_tail(self, path, max_lines=40):
        if not path or not os.path.exists(path):
            return ""
        try:
            with open(path, "r") as f:
                lines = f.readlines()
            return "".join(lines[-max_lines:])
        except Exception:
            return ""

    def _wait_server_ready(self, timeout_secs=120):
        if self.api_port is None:
            raise RuntimeError("API port is not set")

        deadline = time.time() + timeout_secs
        health_url = f"http://127.0.0.1:{self.api_port}/health"
        while time.time() < deadline:
            if self.server_process and self.server_process.poll() is not None:
                stderr_tail = self._read_log_tail(
                    os.path.join(globals.get_results_dir(), f"imagegen_server_{self.api_port}_stderr.log")
                )
                raise RuntimeError(
                    f"ImageGenServer process exited early on port {self.api_port} "
                    f"(rc={self.server_process.returncode}).\n"
                    f"stderr tail:\n{stderr_tail}"
                )
            try:
                with urllib.request.urlopen(health_url, timeout=2) as resp:
                    if resp.status == 200:
                        return
            except Exception:
                time.sleep(1)

        stderr_tail = self._read_log_tail(
            os.path.join(globals.get_results_dir(), f"imagegen_server_{self.api_port}_stderr.log")
        )
        raise TimeoutError(
            f"ImageGenServer did not become healthy on port {self.api_port}.\n"
            f"stderr tail:\n{stderr_tail}"
        )

    def _post_json(self, url, payload, timeout=120):
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read().decode("utf-8")
        return json.loads(data)

    def run_setup(self, *args, **kwargs):
        print("ImageGenServer setup")

        model = kwargs.get("model", self.get_default_config()["model"])
        device = kwargs.get("device", self.get_default_config()["device"])
        mps = int(kwargs.get("mps", self.get_default_config()["mps"]))
        api_port = int(kwargs.get("api_port", self.get_default_config()["api_port"]))
        scheduler = kwargs.get("scheduler", self.get_default_config()["scheduler"])
        priority = int(kwargs.get("priority", self.get_default_config()["priority"]))

        scheduler = scheduler.lower() if isinstance(scheduler, str) else None
        if scheduler not in {None, "naive", "tally", "tgs"}:
            raise ValueError(f"Unsupported scheduler for ImageGenServer: {scheduler}")

        self._setup_scheduler = scheduler
        self.api_port = api_port

        server_script = os.path.join(repo_dir, "applications", "ImageGenServer", "backend.py")
        server_cmd = [
            sys.executable,
            "-u",
            server_script,
            "--host",
            "127.0.0.1",
            "--port",
            str(api_port),
            "--model",
            model,
            "--device",
            device,
            "--mps",
            str(mps),
        ]

        env = os.environ.copy()
        env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(mps)
        env["PRIORITY"] = str(priority)
        launch_cmd = server_cmd

        results_dir = globals.get_results_dir()
        os.makedirs(results_dir, exist_ok=True)
        self.server_stdout = open(os.path.join(results_dir, f"imagegen_server_{api_port}_stdout.log"), "w")
        self.server_stderr = open(os.path.join(results_dir, f"imagegen_server_{api_port}_stderr.log"), "w")

        self.server_process = subprocess.Popen(
            launch_cmd,
            stdout=self.server_stdout,
            stderr=self.server_stderr,
            text=True,
            env=env,
            start_new_session=True,
        )

        try:
            self._wait_server_ready(timeout_secs=360)
        except Exception:
            self._stop_server_process()

            stderr_path = os.path.join(results_dir, f"imagegen_server_{api_port}_stderr.log")
            stderr_tail = self._read_log_tail(stderr_path)
            raise RuntimeError(
                f"ImageGenServer setup failed on port {api_port}. "
                f"stderr log: {stderr_path}\n"
                f"stderr tail:\n{stderr_tail}"
            )

        print("ImageGenServer setup complete")
        return {
            "status": "setup_complete",
            "config": self.config,
            "api_port": api_port,
            "scheduler": scheduler,
            "priority": priority,
            "mps": mps,
        }

    def run_cleanup(self, *args, **kwargs):
        print("ImageGenServer cleanup")

        self._stop_server_process()

        if self.server_stdout:
            self.server_stdout.close()
            self.server_stdout = None
        if self.server_stderr:
            self.server_stderr.close()
            self.server_stderr = None

        return {"status": "cleanup_complete"}

    def run_application(self, *args, **kwargs):
        if self.server_process is None or self.server_process.poll() is not None:
            raise RuntimeError("Server is not running. Call run_setup before run_application.")
        if not self.imagegen_prompts:
            raise RuntimeError("No prompts available. Call load_dataset before run_application.")

        prompt = self.imagegen_prompts.pop(0)
        # Keep runtime knobs fixed to reduce config surface.
        num_inference_steps = 28
        guidance_scale = 3.5
        seed = None

        api_url = f"http://127.0.0.1:{self.api_port}/generate"
        response = self._post_json(
            api_url,
            {
                "prompt": prompt,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "seed": seed,
            },
            timeout=600,
        )

        if "error" in response:
            raise RuntimeError(f"ImageGenServer generation failed: {response['error']}")

        return {
            "status": "image_gen_complete",
            "total time": response.get("total_time"),
        }

    def load_dataset(self, *args, **kwargs):
        ds_imagegen = load_dataset("sentence-transformers/coco-captions")
        ds_imagegen = ds_imagegen["train"]
        ds_imagegen = ds_imagegen.shuffle(seed=42)
        ds_imagegen = ds_imagegen.select(range(0, 100))
        for item in ds_imagegen:
            self.imagegen_prompts.append(item["caption1"])

    def get_default_config(self) -> Dict[str, Any]:
        return {
            "model": "<MODELS_DIR>/stable-diffusion-3.5-medium-turbo",
            "device": "gpu",
            "mps": 100,
            "api_port": 8020,
            "scheduler": None,
            "priority": 1,
        }
