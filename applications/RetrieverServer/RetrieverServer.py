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


class RetrieverServer(Application):
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
                    os.path.join(globals.get_results_dir(), f"retrieverserver_{self.api_port}_stderr.log")
                )
                raise RuntimeError(
                    f"RetrieverServer process exited early on port {self.api_port} "
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
            os.path.join(globals.get_results_dir(), f"retrieverserver_{self.api_port}_stderr.log")
        )
        raise TimeoutError(
            f"RetrieverServer did not become healthy on port {self.api_port}.\n"
            f"stderr tail:\n{stderr_tail}"
        )

    def _post_json(self, url, payload, timeout=120):
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read().decode("utf-8")
        return json.loads(data)

    def _resolve_path(self, value):
        value = str(value)
        if os.path.isabs(value):
            return value
        repo_relative = os.path.join(repo_dir, value)
        if os.path.exists(repo_relative):
            return repo_relative
        return os.path.abspath(value)

    def _ensure_retriever_index(self, index_path, docs_path, corpus_path, auto_build_index):
        if os.path.exists(index_path) and os.path.exists(docs_path):
            return

        if auto_build_index:
            retriever_dir = os.path.join(repo_dir, "applications", "Retriever")
            build_script = os.path.join(retriever_dir, "build_index.py")
            expected_corpus = os.path.join(retriever_dir, "ragqa_arena_tech_corpus.jsonl")

            if os.path.exists(build_script) and os.path.exists(expected_corpus):
                print("RetrieverServer: index missing, running build_index.py ...")
                subprocess.run([sys.executable, build_script], cwd=retriever_dir, check=True)

        if os.path.exists(index_path) and os.path.exists(docs_path):
            return

        raise FileNotFoundError(
            "RetrieverServer index files are missing.\n"
            f"Expected index: {index_path}\n"
            f"Expected docs:  {docs_path}\n"
            f"Corpus looked for: {corpus_path}\n"
            "To fix on the target machine, ensure applications/Retriever assets are present and run:\n"
            "  cd <WORKSPACE>/ConsumerBench/applications/Retriever\n"
            "  python build_index.py"
        )

    def run_setup(self, *args, **kwargs):
        print("RetrieverServer setup")

        api_port = int(kwargs.get("api_port", self.get_default_config()["api_port"]))
        index_path = self._resolve_path(kwargs.get("index_path", self.get_default_config()["index_path"]))
        docs_path = self._resolve_path(kwargs.get("docs_path", self.get_default_config()["docs_path"]))
        model_name = kwargs.get("model_name", self.get_default_config()["model_name"])
        default_k = int(kwargs.get("default_k", self.get_default_config()["default_k"]))
        device = kwargs.get("device", self.get_default_config()["device"])
        mps = int(kwargs.get("mps", self.get_default_config()["mps"]))
        corpus_path = self._resolve_path(
            kwargs.get("corpus_path", "applications/Retriever/ragqa_arena_tech_corpus.jsonl")
        )
        auto_build_index = bool(kwargs.get("auto_build_index", True))
        scheduler = kwargs.get("scheduler", self.get_default_config()["scheduler"])
        priority = int(kwargs.get("priority", self.get_default_config()["priority"]))

        scheduler = scheduler.lower() if isinstance(scheduler, str) else None
        if scheduler not in {None, "naive", "tally", "tgs"}:
            raise ValueError(f"Unsupported scheduler for RetrieverServer: {scheduler}")

        self._setup_scheduler = scheduler
        self.api_port = api_port

        self._ensure_retriever_index(index_path, docs_path, corpus_path, auto_build_index)

        server_script = os.path.join(repo_dir, "applications", "RetrieverServer", "backend.py")
        server_cmd = [
            sys.executable,
            "-u",
            server_script,
            "--host",
            "127.0.0.1",
            "--port",
            str(api_port),
            "--index-path",
            str(index_path),
            "--docs-path",
            str(docs_path),
            "--model-name",
            str(model_name),
            "--default-k",
            str(default_k),
            "--device",
            str(device),
            "--mps",
            str(mps),
        ]

        env = os.environ.copy()
        env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(mps)
        env["PRIORITY"] = str(priority)

        results_dir = globals.get_results_dir()
        os.makedirs(results_dir, exist_ok=True)
        self.server_stdout = open(os.path.join(results_dir, f"retrieverserver_{api_port}_stdout.log"), "w")
        self.server_stderr = open(os.path.join(results_dir, f"retrieverserver_{api_port}_stderr.log"), "w")

        self.server_process = subprocess.Popen(
            server_cmd,
            stdout=self.server_stdout,
            stderr=self.server_stderr,
            text=True,
            env=env,
            start_new_session=True,
        )

        try:
            self._wait_server_ready(timeout_secs=180)
        except Exception:
            self._stop_server_process()

            stderr_path = os.path.join(results_dir, f"retrieverserver_{api_port}_stderr.log")
            stderr_tail = self._read_log_tail(stderr_path)
            raise RuntimeError(
                f"RetrieverServer setup failed on port {api_port}. "
                f"stderr log: {stderr_path}\n"
                f"stderr tail:\n{stderr_tail}"
            )

        print("RetrieverServer setup complete")
        return {
            "status": "setup_complete",
            "config": self.config,
            "api_port": api_port,
            "scheduler": scheduler,
            "priority": priority,
            "mps": mps,
        }

    def run_cleanup(self, *args, **kwargs):
        print("RetrieverServer cleanup")

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

        query = kwargs.get("query", self.get_default_config()["query"])
        k = int(kwargs.get("k", self.get_default_config()["default_k"]))

        api_url = f"http://127.0.0.1:{self.api_port}/retrieve"
        response = self._post_json(
            api_url,
            {
                "query": query,
                "k": k,
            },
            timeout=300,
        )

        if "error" in response:
            raise RuntimeError(f"RetrieverServer retrieval failed: {response['error']}")

        return {
            "status": "retrieval_complete",
            "query": response.get("query", query),
            "passages": response.get("passages", []),
            "total time": response.get("total_time"),
        }

    def load_dataset(self, *args, **kwargs):
        print("RetrieverServer loading dataset")
        return {"status": "dataset_loaded"}

    def get_default_config(self) -> Dict[str, Any]:
        return {
            "api_port": 8030,
            "index_path": f"{repo_dir}/applications/Retriever/rag_index/corpus.faiss",
            "docs_path": f"{repo_dir}/applications/Retriever/rag_index/docs.jsonl",
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "default_k": 5,
            "device": "cpu",
            "mps": 100,
            "scheduler": None,
            "priority": 1,
            "query": "What is retrieval augmented generation?",
        }
