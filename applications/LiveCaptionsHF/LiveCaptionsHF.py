import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from typing import Any, Dict

import src.globals as globals

repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(repo_dir)

from applications.application import Application


class LiveCaptionsHF(Application):
    def __init__(self):
        super().__init__()
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
                # Kill the process group because tally wrapping may spawn shell children.
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

    def _wait_server_ready(self, timeout_secs=180):
        if self.api_port is None:
            raise RuntimeError("API port is not set")

        deadline = time.time() + timeout_secs
        health_url = f"http://127.0.0.1:{self.api_port}/health"
        while time.time() < deadline:
            if self.server_process and self.server_process.poll() is not None:
                stderr_tail = self._read_log_tail(
                    os.path.join(globals.get_results_dir(), f"livecaptionshf_server_{self.api_port}_stderr.log")
                )
                raise RuntimeError(
                    f"LiveCaptionsHF server exited early on port {self.api_port} "
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
            os.path.join(globals.get_results_dir(), f"livecaptionshf_server_{self.api_port}_stderr.log")
        )
        raise TimeoutError(
            f"LiveCaptionsHF server did not become healthy on port {self.api_port}.\n"
            f"stderr tail:\n{stderr_tail}"
        )

    def _post_json(self, url, payload, timeout=180):
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read().decode("utf-8")
        return json.loads(data)

    def _resolve_audio_path(self, audio_file):
        if os.path.isabs(audio_file):
            return audio_file

        repo_relative = os.path.join(repo_dir, audio_file)
        if os.path.exists(repo_relative):
            return repo_relative

        return os.path.abspath(audio_file)

    def run_setup(self, *args, **kwargs):
        print("LiveCaptionsHF setup")

        model_id = kwargs.get("model", self.get_default_config()["model"])
        device = kwargs.get("device", self.get_default_config()["device"])
        mps = int(kwargs.get("mps", self.get_default_config()["mps"]))
        api_port = int(kwargs.get("api_port", self.get_default_config()["api_port"]))
        torch_dtype_name = "float16"
        trust_remote_code = False
        hf_token = os.environ.get("HF_TOKEN")
        scheduler = kwargs.get("scheduler", self.get_default_config()["scheduler"])
        priority = int(kwargs.get("priority", self.get_default_config()["priority"]))

        scheduler = scheduler.lower() if isinstance(scheduler, str) else None
        if scheduler not in {None, "naive", "tally", "tgs"}:
            raise ValueError(f"Unsupported scheduler for LiveCaptionsHF: {scheduler}")

        self._setup_scheduler = scheduler
        self.api_port = api_port

        server_script = os.path.join(repo_dir, "applications", "LiveCaptionsHF", "backend.py")
        server_cmd = [
            sys.executable,
            "-u",
            server_script,
            "--host",
            "127.0.0.1",
            "--port",
            str(api_port),
            "--model",
            model_id,
            "--device",
            device,
            "--torch-dtype",
            str(torch_dtype_name),
        ]

        if trust_remote_code:
            server_cmd.append("--trust-remote-code")
        if hf_token:
            server_cmd.extend(["--hf-token", hf_token])

        env = os.environ.copy()
        env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(mps)
        env["PRIORITY"] = str(priority)
        launch_cmd = server_cmd

        results_dir = globals.get_results_dir()
        os.makedirs(results_dir, exist_ok=True)
        self.server_stdout = open(os.path.join(results_dir, f"livecaptionshf_server_{api_port}_stdout.log"), "w")
        self.server_stderr = open(os.path.join(results_dir, f"livecaptionshf_server_{api_port}_stderr.log"), "w")

        self.server_process = subprocess.Popen(
            launch_cmd,
            stdout=self.server_stdout,
            stderr=self.server_stderr,
            text=True,
            env=env,
            start_new_session=True,
        )

        try:
            self._wait_server_ready(timeout_secs=480)
        except Exception:
            self._stop_server_process()

            stderr_path = os.path.join(results_dir, f"livecaptionshf_server_{api_port}_stderr.log")
            stderr_tail = self._read_log_tail(stderr_path)
            raise RuntimeError(
                f"LiveCaptionsHF setup failed on port {api_port}. "
                f"stderr log: {stderr_path}\n"
                f"stderr tail:\n{stderr_tail}"
            )

        print("LiveCaptionsHF setup complete")
        return {
            "status": "setup_complete",
            "config": self.config,
            "api_port": api_port,
            "scheduler": scheduler,
            "priority": priority,
            "mps": mps,
        }

    def run_cleanup(self, *args, **kwargs):
        print("LiveCaptionsHF cleanup")

        self._stop_server_process()

        if self.server_stdout:
            self.server_stdout.close()
            self.server_stdout = None
        if self.server_stderr:
            self.server_stderr.close()
            self.server_stderr = None

        return {"status": "cleanup_complete"}

    def run_application(self, *args, **kwargs):
        print("LiveCaptionsHF application")

        if self.server_process is None or self.server_process.poll() is not None:
            raise RuntimeError("Server is not running. Call run_setup before run_application.")

        audio_file = kwargs.get("client_command_file", self.get_default_config()["client_command_file"])
        audio_file = self._resolve_audio_path(str(audio_file))

        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"Audio file not found for LiveCaptionsHF: {audio_file}")

        # Hardcoded LiveCaptionsHF behavior to keep config surface minimal.
        language = "en"
        task = "transcribe"
        chunk_length_s = 30.0
        batch_size = 8
        use_realtime_chunking = True
        chunk_seconds = 2.0
        simulate_realtime = True
        return_full_text = False
        max_text_chars = 320
        include_segments = False

        if use_realtime_chunking:
            api_url = f"http://127.0.0.1:{self.api_port}/transcribe_realtime"
            response = self._post_json(
                api_url,
                {
                    "audio_file": audio_file,
                    "language": language,
                    "task": task,
                    "chunk_seconds": chunk_seconds,
                    "simulate_realtime": simulate_realtime,
                },
                timeout=1800,
            )
        else:
            api_url = f"http://127.0.0.1:{self.api_port}/transcribe"
            response = self._post_json(
                api_url,
                {
                    "audio_file": audio_file,
                    "language": language,
                    "task": task,
                    "chunk_length_s": chunk_length_s,
                    "batch_size": batch_size,
                },
                timeout=1800,
            )

        if "error" in response:
            raise RuntimeError(f"LiveCaptionsHF transcription failed: {response['error']}")

        text_value = response.get("text")
        if isinstance(text_value, str) and not return_full_text and max_text_chars > 0 and len(text_value) > max_text_chars:
            text_value = text_value[:max_text_chars].rstrip() + "..."

        result = {
            "status": "live_captions_complete",
            "text": text_value,
            "audio_duration": response.get("audio_duration"),
            "processing_time": response.get("processing_time"),
            "total time": response.get("processing_time"),
            "use_realtime_chunking": bool(use_realtime_chunking),
        }

        chunk_entries = response.get("chunks")
        if isinstance(chunk_entries, list) and chunk_entries:
            result["num_chunks"] = len(chunk_entries)
            result["segment_count"] = sum(
                len(chunk.get("segments", [])) for chunk in chunk_entries if isinstance(chunk, dict)
            )
            for chunk in chunk_entries:
                idx = chunk.get("chunk_index")
                ptime = chunk.get("processing_time")
                if isinstance(idx, int) and ptime is not None:
                    result[f"processing time_chunk_{idx}"] = ptime

            if include_segments:
                result["segments"] = chunk_entries
        else:
            result["processing time_chunk_0"] = response.get("processing_time")
            segments = response.get("segments")
            if isinstance(segments, list):
                result["segment_count"] = len(segments)
                if include_segments:
                    result["segments"] = segments

        return result

    def load_dataset(self, *args, **kwargs):
        """No dataset preload required; run_application uses configured wav paths."""

    def get_default_config(self) -> Dict[str, Any]:
        return {
            "model": "openai/whisper-large-v3-turbo",
            "device": "gpu",
            "mps": 100,
            "api_port": 5010,
            "client_command_file": "applications/LiveCaptions/whisper-earnings21/4320211_chunk_001.wav",
            "scheduler": None,
            "priority": 1,
        }
