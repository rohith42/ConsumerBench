## Add DeepResearch class here
import time
from typing import Any, Dict
import sys
import os
# from datasets import load_dataset
import subprocess
import requests
import json

repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(repo_dir)

from applications.application import Application
# import src.utils as utils
# import src.globals as globals
# from inference_backends.Llamacpp import LlamaCpp
# from inference_backends.Vllm import Vllm

# pasted from /local1/samarjit/workspace/TGS/scripts/test_text_inference_tgs.sh:
# i basically want this python script to be run when the application starts.
# #!/usr/bin/env bash
# set -o errexit
# set -o pipefail
# set -o nounset
# set -o xtrace

# pushd "$(dirname "$0")/.." >/dev/null
# # mirror test_tgs.sh: capture stdout/stderr and save to backup_logs/test_tgs.log
# # mkdir -p backup_logs
# python worker.py --trace config/test_text_inference_tgs.csv --log_path results/test_text_inference_results.csv --gpus 0 2>&1 | tee backup_logs/test_tgs.log

# popd >/dev/null


class Dummy(Application):
    def __init__(self):
        super().__init__()
        self.chatbot_prompts = []
        self.backend = None

    def run_setup(self, *args, **kwargs):
        print("Chatbot setup")
        api_port = kwargs.get('api_port', self.get_default_config()['api_port'])
        model = kwargs.get('model', self.get_default_config()['model'])
        device = kwargs.get('device', self.get_default_config()['device'])
        mps = kwargs.get('mps', self.get_default_config()['mps'])
        backend_type = kwargs.get('backend', self.get_default_config()['backend'])

        print(f"Dummy setup complete")

        return {"status": "setup_complete", "config": self.config}

    def run_cleanup(self, *args, **kwargs):
        print("Chatbot cleanup")
        return {"status": "cleanup_complete"}

    def run_application(self, *args, **kwargs):
        print(f"Chatbot application")
        # api_port = kwargs.get('api_port', self.get_default_config()['api_port'])
        model = kwargs.get('model', self.get_default_config()['model'])

        chatbot_prompt = self.chatbot_prompts.pop(0)
        chatbot_prompts = [chatbot_prompt]

        ttft = None
        token_count = None
        first_token_time = None

        start_time = time.time()

        # just some testing: print where we are, and what venv we are using, and sleep for 1 second to simulate inference time
        print(f"Running dummy application with model: {model}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Using Python executable: {sys.executable}")
        time.sleep(1)
        # we want to enter TGS and run the test_text_inference_tgs.sh script.
        # but we will have to launch a venv and run the script inside the venv. let's assume the venv is located at /local1/samarjit/workspace/TGS/.venv
        tgs_venv_path = "/local1/samarjit/workspace/TGS/.venv"
        tgs_script_path = "/local1/samarjit/workspace/TGS/scripts/test_text_inference_tgs.sh"
        # activate the venv and run the script, and capture the output
        command = f"source {tgs_venv_path}/bin/activate && bash {tgs_script_path}"
        print(f"Running command: {command}")
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        print(f"Command output: {result.stdout}")
        print(f"Command error: {result.stderr}")

        for prompt in chatbot_prompts:
            # just sleep for 1 second to simulate inference time
            time.sleep(1)

        # make up some token count and first token time
        token_count = 100
        first_token_time = start_time + 0.5

        end_time = time.time()
        print(f"Total time: {end_time - start_time:.4f} seconds")
        print(f"Completion tokens: {token_count}")

        print(f"{end_time-first_token_time}, token counts: {token_count}")
        tpot = (end_time - first_token_time) / token_count if token_count else None
        itl = (end_time - start_time) / token_count if token_count else None

        return {"status": "chatbot_complete", "ttft": ttft, "tpot": tpot, "itl": itl}

    def load_dataset(self, *args, **kwargs):
        """Load the chatbot dataset"""
        # append some dummy prompts to self.chatbot_prompts
        self.chatbot_prompts.append("What is the capital of France?")
        self.chatbot_prompts.append("What is the largest mammal?")
        self.chatbot_prompts.append("What is the meaning of life?")
        print(f"Loaded dataset with {len(self.chatbot_prompts)} prompts")

    def get_default_config(self) -> Dict[str, Any]:
        return {
            "model": f"{repo_dir}/models/Llama-3.2-3B-Instruct-GGUF/Llama-3.2-3B-Instruct-f16.gguf",
            "device": "gpu",
            "mps": 100,
            "api_port": 8080,
            "llamacpp_path": f"{repo_dir}/inference_backends/llama.cpp",
            "dataset": f"lmsys/lmsys-chat-1m",
            "backend": "llamacpp",
            "vllm_path": f"{repo_dir}/inference_backends/vllm"
        }
    