## Add DeepResearch class here
import time
from typing import Any, Dict
import sys
import os
from datasets import load_dataset
import glob
import subprocess
import requests
import json

repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(repo_dir)

TGS_PATH = os.getenv('TGS_PATH', "/local1/rohithl/TGS")
TGS_RATE_MULTIPLIER_PATH = os.path.join(TGS_PATH, "gsharing", "tpot_multiplier.txt")
TGS_RATE_MULTIPLIER_UPDATE_EVERY = 5
TGS_RATE_MULTIPLIER_DISABLED = -1.0

from applications.application import Application
import src.utils as utils
import src.globals as globals
from inference_backends.Llamacpp import LlamaCpp
from inference_backends.Vllm import Vllm
from inference_backends.TGSLlamaCpp import TGSLlamaCpp

class Chatbot(Application):
    def __init__(self):
        super().__init__()
        self.chatbot_prompts = []
        self.backend = None
        self.tgs_slo_seconds = None
        self.enable_tgs_multiplier_updates = False
        self._line_count = 0

    @staticmethod
    def _write_rate_multiplier(self, multiplier: float, tpot: float, slo_seconds: float) -> None:
        output_dir = os.path.dirname(TGS_RATE_MULTIPLIER_PATH)
        os.makedirs(output_dir, exist_ok=True)

        temp_path = f"{TGS_RATE_MULTIPLIER_PATH}.tmp"
        payload = f"{multiplier:.8f}\n"
        with open(temp_path, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, TGS_RATE_MULTIPLIER_PATH)
        if self._line_count % 20 == 0:
            print(
                f"Wrote TGS rate multiplier {multiplier:.6f} from tpot={tpot:.6f}s "
                f"(slo={slo_seconds:.6f}s) to {TGS_RATE_MULTIPLIER_PATH}"
            )

    @staticmethod
    def _write_disabled_rate_multiplier() -> None:
        output_dir = os.path.dirname(TGS_RATE_MULTIPLIER_PATH)
        os.makedirs(output_dir, exist_ok=True)
        with open(TGS_RATE_MULTIPLIER_PATH, "w", encoding="utf-8") as handle:
            handle.write(f"{TGS_RATE_MULTIPLIER_DISABLED:.8f}\n")
        print(f"Wrote disabled TGS rate multiplier to {TGS_RATE_MULTIPLIER_PATH}")

    @staticmethod
    def _extract_stream_text(data: Dict[str, Any]) -> str:
        choices = data.get("choices") or []
        for choice in choices:
            delta = choice.get("delta") or {}
            message = choice.get("message") or {}
            text = delta.get("content") or choice.get("text") or message.get("content")
            if text:
                return text
        return ""

    def _maybe_update_rate_multiplier(
        self,
        current_time: float,
        first_token_time: float,
        token_count: int,
        estimated_token_count: int,
        last_multiplier_update_count: int,
        last_multiplier_update_time: float = None,
    ) -> tuple:
        """Update the TGS rate multiplier using TPOT measured since the last update.

        Returns a tuple `(new_last_multiplier_update_count, new_last_multiplier_update_time)`.
        """
        if not self.enable_tgs_multiplier_updates or first_token_time is None:
            return last_multiplier_update_count, last_multiplier_update_time

        observed_tokens = token_count if token_count is not None else estimated_token_count
        if observed_tokens <= 0:
            return last_multiplier_update_count, last_multiplier_update_time

        # tokens observed since the last multiplier update
        if last_multiplier_update_count and observed_tokens > last_multiplier_update_count:
            delta_tokens = observed_tokens - last_multiplier_update_count
        else:
            delta_tokens = observed_tokens

        if delta_tokens <= 0:
            return last_multiplier_update_count, last_multiplier_update_time

        # elapsed time since the last multiplier update (or since first token)
        if last_multiplier_update_time is not None:
            delta_time = current_time - last_multiplier_update_time
        else:
            delta_time = current_time - first_token_time

        observed_tpot = delta_time / max(1, delta_tokens)
        if observed_tpot <= 0:
            return last_multiplier_update_count, last_multiplier_update_time

        multiplier = self.tgs_slo_seconds / observed_tpot
        should_update = (
            last_multiplier_update_count == 0 or
            token_count is not None or
            observed_tokens - last_multiplier_update_count >= TGS_RATE_MULTIPLIER_UPDATE_EVERY
        )
        if not should_update:
            return last_multiplier_update_count, last_multiplier_update_time

        try:
            self._write_rate_multiplier(self, multiplier, observed_tpot, self.tgs_slo_seconds)
            return observed_tokens, current_time
        except Exception as exc:
            print(f"Failed to write TGS rate multiplier file: {exc}")
            return last_multiplier_update_count, last_multiplier_update_time
    def run_setup(self, *args, **kwargs):
        print("Chatbot setup")
        api_port = kwargs.get('api_port', self.get_default_config()['api_port'])
        model = kwargs.get('model', self.get_default_config()['model'])
        device = kwargs.get('device', self.get_default_config()['device'])
        mps = kwargs.get('mps', self.get_default_config()['mps'])
        backend_type = kwargs.get('backend', self.get_default_config()['backend'])
        if 'tgs_slo' in kwargs and kwargs.get('tgs_slo') is not None:
            self.tgs_slo_seconds = float(kwargs.get('tgs_slo'))
            self.enable_tgs_multiplier_updates = True
            print(f"TGS multiplier updates enabled with slo={self.tgs_slo_seconds:.6f}s")
        else:
            self.tgs_slo_seconds = None
            self.enable_tgs_multiplier_updates = False
            print("TGS multiplier updates disabled (no tgs_slo provided)")

        if backend_type == 'tgs-llamacpp' and not self.enable_tgs_multiplier_updates:
            self._write_disabled_rate_multiplier()

        if backend_type == 'vllm':
            self.backend = Vllm()
            vllm_path = kwargs.get('vllm_path', self.get_default_config()['vllm_path'])
            self.backend.launch_backend(api_port=api_port, model=model, device=device, vllm_path=vllm_path)
        elif backend_type == 'tgs-llamacpp':
            self.backend = TGSLlamaCpp()
            self.backend.launch_backend(
                api_port=api_port,
                priority=kwargs.get('tgs_priority', 'high'),
                tgs_path=kwargs.get('tgs_path', self.get_default_config()['tgs_path']),
                model=kwargs.get('tgs_model'),
            )
        else:
            self.backend = LlamaCpp()
            llamacpp_path = kwargs.get('llamacpp_path', self.get_default_config()['llamacpp_path'])
            self.backend.launch_backend(api_port=api_port, model=model, device=device, mps=mps, llamacpp_path=llamacpp_path)

        print(f"Chatbot setup complete")

        return {"status": "setup_complete", "config": self.config}

    def run_cleanup(self, *args, **kwargs):
        print("Chatbot cleanup")
        api_port = kwargs.get('api_port', self.get_default_config()['api_port'])

        self.backend.cleanup_backend(api_port=api_port, priority=kwargs.get('tgs_priority', 'high'))
        return {"status": "cleanup_complete"}

    def run_application(self, *args, **kwargs):
        print(f"Chatbot application")
        api_port = kwargs.get('api_port', self.get_default_config()['api_port'])
        model = kwargs.get('model', self.get_default_config()['model'])

        chatbot_prompt = self.chatbot_prompts.pop(0)

        if isinstance(chatbot_prompt, dict) and "messages" in chatbot_prompt:
            api_url = f"http://127.0.0.1:{api_port}/v1/chat/completions"
            params = chatbot_prompt.get("parameters", {})
            payload = {
                "model": model,
                "messages": chatbot_prompt["messages"],
                "temperature": params.get("temperature", 0.0),
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            if params.get("stop"):
                payload["stop"] = params["stop"]
            if params.get("max_tokens") is not None:
                payload["max_tokens"] = params["max_tokens"]
        else:
            api_url = f"http://127.0.0.1:{api_port}/v1/completions"
            payload = {
                "model": model,
                "prompt": chatbot_prompt,
                "max_tokens": 256,
                "temperature": 0,
                "top_p": 0.9,
                "seed": 141293,
                "stream": True,
                "stream_options": {"include_usage": True},
            }

        ttft = None
        token_count = None
        first_token_time = None
        estimated_text_chars = 0
        last_multiplier_update_count = 0
        last_multiplier_update_time = None
        tpot_estimates = {"elapsed_seconds": [], "tpot": []}
        last_recorded_estimated_tokens = 0

        start_time = time.time()

        headers = {
            "Content-Type": "application/json"
        }

        try:
            with requests.post(api_url, json=payload, headers=headers, stream=True) as response:
                if response.status_code != 200:
                    print("HTTP Error:", response.status_code, response.text)
                    return
                self._line_count = 0

                for line in response.iter_lines(decode_unicode=True):
                    if line:
                        self._line_count += 1
                        current_time = time.time()
                        if ttft is None:
                            ttft = current_time - start_time
                            first_token_time = current_time
                            print(f"Time to first token: {ttft:.4f} seconds")

                        try:
                            clean_line = line.strip().replace("data: ", "")
                            if clean_line == "[DONE]":
                                break

                            data = json.loads(clean_line)

                            # Capture usage from any chunk that has it (null-safe).
                            # vLLM sends usage in a separate final chunk after finish_reason;
                            # llamacpp includes it in the finish_reason chunk.
                            usage = data.get("usage") or {}
                            if usage.get("completion_tokens") is not None:
                                token_count = usage["completion_tokens"]

                            stream_text = self._extract_stream_text(data)
                            if stream_text:
                                estimated_text_chars += len(stream_text)

                            estimated_token_count = max(1, estimated_text_chars // 4) if estimated_text_chars > 0 else 0
                            observed_token_count = token_count if token_count is not None else estimated_token_count
                            if first_token_time is not None and observed_token_count > 0 and observed_token_count > last_recorded_estimated_tokens:
                                elapsed_seconds = current_time - first_token_time
                                tpot_estimates["elapsed_seconds"].append(elapsed_seconds)
                                tpot_estimates["tpot"].append(elapsed_seconds / observed_token_count)
                                last_recorded_estimated_tokens = observed_token_count

                            last_multiplier_update_count, last_multiplier_update_time = self._maybe_update_rate_multiplier(
                                current_time,
                                first_token_time,
                                token_count,
                                estimated_token_count,
                                last_multiplier_update_count,
                                last_multiplier_update_time,
                            )

                        except json.JSONDecodeError:
                            continue

        except Exception as e:
            print("Request failed:", e)

        end_time = time.time()
        final_estimated_token_count = max(1, estimated_text_chars // 4) if estimated_text_chars > 0 else 0
        final_token_count = token_count if token_count is not None else final_estimated_token_count
        if first_token_time is not None and final_token_count > 0 and final_token_count != last_recorded_estimated_tokens:
            elapsed_seconds = end_time - first_token_time
            tpot_estimates["elapsed_seconds"].append(elapsed_seconds)
            tpot_estimates["tpot"].append(elapsed_seconds / final_token_count)
            last_recorded_estimated_tokens = final_token_count

        last_multiplier_update_count, last_multiplier_update_time = self._maybe_update_rate_multiplier(
            end_time,
            first_token_time,
            token_count,
            final_estimated_token_count,
            last_multiplier_update_count,
            last_multiplier_update_time,
        )
        print(f"Total time: {end_time - start_time:.4f} seconds")
        print(f"Completion tokens: {token_count}")

        if first_token_time is not None:
            print(f"{end_time-first_token_time}, token counts: {token_count}")
        else:
            print(f"No first token time recorded, token counts: {token_count}")

        tpot = (end_time - first_token_time) / final_token_count if first_token_time is not None and final_token_count else None
        itl = (end_time - start_time) / final_token_count if final_token_count else None

        return {
            "status": "chatbot_complete",
            "ttft": ttft,
            "tpot": tpot,
            "tpot_estimates": tpot_estimates,
            "itl": itl,
            "completion_tokens": token_count,
        }

    def load_dataset(self, *args, **kwargs):
        """Load the chatbot dataset"""
        mcp_trace = kwargs.get('mcp_trace_json', None)
        dataset_source = kwargs.get('dataset', None) or self.get_default_config()['dataset']
        resolved = dataset_source if os.path.isabs(dataset_source) else os.path.join(repo_dir, dataset_source)

        if mcp_trace is not None:
            trace_json = json.loads(open(mcp_trace, 'r').read())
            for section_name, section_data in trace_json.items():
                if section_name == "text_generate":
                    for call_id, call_data in section_data.items():
                        prompt = call_data.get('prompt', None)
                        if prompt is not None:
                            self.chatbot_prompts.append(prompt)
        elif os.path.isdir(resolved):
            for jf in sorted(glob.glob(os.path.join(resolved, "request_*.json"))):
                with open(jf) as f:
                    data = json.load(f)
                self.chatbot_prompts.append({
                    "messages": data.get("messages", []),
                    "parameters": data.get("parameters", {}),
                })
        else:
            ds_textgen = load_dataset(dataset_source)
            ds_textgen = ds_textgen["train"]
            ds_textgen = ds_textgen.shuffle(seed=42)
            ds_textgen = ds_textgen.select(range(0, 100))
            for item in ds_textgen:
                self.chatbot_prompts.append(item['conversation'][0]['content'])

    def get_default_config(self) -> Dict[str, Any]:
        return {
            "model": f"{repo_dir}/models/Llama-3.2-3B-Instruct-GGUF/Llama-3.2-3B-Instruct-f16.gguf",
            "device": "gpu",
            "mps": 100,
            "api_port": 8080,
            "llamacpp_path": f"{repo_dir}/inference_backends/llama.cpp",
            "dataset": f"lmsys/lmsys-chat-1m",
            "backend": "llamacpp",
            "vllm_path": f"{repo_dir}/inference_backends/vllm",
            "tgs_path": TGS_PATH,
        }
    