import argparse
import json
import logging
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


class ChatbotHFServer:
    def __init__(self, model_id, device, torch_dtype_name, trust_remote_code=False, hf_token=None):
        self.model_id = model_id
        self.device = self._resolve_device(device)
        self.torch_dtype = self._resolve_dtype(torch_dtype_name)

        logging.info(
            "Initializing ChatbotHFServer model=%s requested_device=%s resolved_device=%s dtype=%s",
            model_id,
            device,
            self.device,
            torch_dtype_name,
        )

        load_start = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            token=hf_token,
            trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=self.torch_dtype,
            token=hf_token,
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=False,
        )
        self.model.to(self.device)
        self.model.eval()
        logging.info("Model and tokenizer loaded in %.2fs", time.time() - load_start)

    def _resolve_device(self, device):
        if device == "gpu":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if device == "mps":
            return "mps" if torch.backends.mps.is_available() else "cpu"
        if device in ["cuda", "cpu", "mps"]:
            return device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _resolve_dtype(self, torch_dtype_name):
        dtype_map = {
            "auto": "auto",
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        return dtype_map.get(torch_dtype_name, "auto")

    def generate(self, prompt, max_new_tokens=215, temperature=0.0, top_p=0.9, seed=141293):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_error = {"error": None}

        generation_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "temperature": temperature,
            "top_p": top_p,
            "pad_token_id": self.tokenizer.eos_token_id,
            "streamer": streamer,
        }

        def _run_generate():
            try:
                with torch.no_grad():
                    self.model.generate(**generation_kwargs)
            except Exception as e:
                generation_error["error"] = e

        start_time = time.time()
        generation_thread = Thread(target=_run_generate, daemon=True)
        generation_thread.start()

        first_token_time = None
        generated_text_parts = []
        for chunk in streamer:
            now = time.time()
            if first_token_time is None and chunk:
                first_token_time = now
            generated_text_parts.append(chunk)

        generation_thread.join()
        if generation_error["error"] is not None:
            raise generation_error["error"]

        end_time = time.time()

        text = "".join(generated_text_parts)
        token_count = len(self.tokenizer.encode(text, add_special_tokens=False))

        return {
            "status": "chatbot_complete",
            "text": text,
            "ttft": (first_token_time - start_time) if first_token_time is not None else None,
            "tpot": ((end_time - first_token_time) / token_count) if token_count > 0 else None,
            "itl": ((end_time - start_time) / token_count) if token_count > 0 else None,
            "total_time": end_time - start_time,
            "token_count": token_count,
        }


def make_handler(app):
    class Handler(BaseHTTPRequestHandler):
        def _write_json(self, status_code, payload):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                logging.info("Health check OK")
                self._write_json(200, {"status": "ok"})
            else:
                self._write_json(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/generate":
                self._write_json(404, {"error": "not found"})
                return

            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw_body = self.rfile.read(length) if length > 0 else b"{}"
                payload = json.loads(raw_body.decode("utf-8"))

                prompt = payload.get("prompt", "")
                if not prompt:
                    self._write_json(400, {"error": "prompt is required"})
                    return

                response = app.generate(
                    prompt=prompt,
                    max_new_tokens=int(payload.get("max_new_tokens", 215)),
                    temperature=float(payload.get("temperature", 0.0)),
                    top_p=float(payload.get("top_p", 0.9)),
                    seed=int(payload.get("seed", 141293)),
                )
                logging.info(
                    "Generation complete token_count=%s total_time=%.3fs ttft=%s",
                    response.get("token_count"),
                    float(response.get("total_time", 0.0)),
                    response.get("ttft"),
                )
                self._write_json(200, response)
            except Exception as e:
                logging.exception("Generation request failed")
                self._write_json(500, {"error": str(e)})

        def log_message(self, format, *args):
            return

    return Handler


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="[chatbothf_server] %(asctime)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--device", type=str, default="gpu")
    parser.add_argument("--torch-dtype", type=str, default="float16")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--hf-token", type=str, default=None)
    args = parser.parse_args()

    app = ChatbotHFServer(
        model_id=args.model,
        device=args.device,
        torch_dtype_name=args.torch_dtype,
        trust_remote_code=args.trust_remote_code,
        hf_token=args.hf_token,
    )

    logging.info("Starting HTTP server on %s:%d", args.host, args.port)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    server.serve_forever()


if __name__ == "__main__":
    main()
