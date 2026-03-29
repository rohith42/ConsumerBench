import argparse
import json
import logging
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from math import gcd

import numpy as np
import soundfile as sf
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

try:
    from scipy.signal import resample_poly
except Exception:
    resample_poly = None


class LiveCaptionsHFServer:
    def __init__(self, model_id, device, torch_dtype_name, trust_remote_code=False, hf_token=None):
        self.model_id = model_id
        self.device = self._resolve_device(device)
        self.torch_dtype = self._resolve_dtype(torch_dtype_name)

        logging.info(
            "Initializing LiveCaptionsHFServer model=%s requested_device=%s resolved_device=%s dtype=%s",
            model_id,
            device,
            self.device,
            torch_dtype_name,
        )

        load_start = time.time()
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            token=hf_token,
            trust_remote_code=trust_remote_code,
        )

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=self.torch_dtype,
            token=hf_token,
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=True,
        )
        self.model.to(self.device)
        self.model.eval()

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )
        logging.info("Whisper model and pipeline loaded in %.2fs", time.time() - load_start)

    def _resolve_device(self, device):
        if device == "gpu":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if device == "mps":
            return "mps" if torch.backends.mps.is_available() else "cpu"
        if device == "cuda":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if device in ["cpu", "mps"]:
            return device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _resolve_dtype(self, torch_dtype_name):
        dtype_map = {
            "auto": "auto",
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        return dtype_map.get(torch_dtype_name, torch.float16)

    def _safe_audio_duration(self, audio_file):
        try:
            info = sf.info(audio_file)
            if info.samplerate > 0:
                return float(info.frames) / float(info.samplerate)
        except Exception:
            return None
        return None

    def _resample_audio(self, audio, src_sr, dst_sr):
        if src_sr == dst_sr:
            return audio

        if resample_poly is not None:
            g = gcd(int(src_sr), int(dst_sr))
            up = int(dst_sr // g)
            down = int(src_sr // g)
            return resample_poly(audio, up, down).astype(np.float32, copy=False)

        # Fallback path when scipy is unavailable.
        src_len = len(audio)
        if src_len == 0:
            return audio
        dst_len = max(1, int(round(src_len * float(dst_sr) / float(src_sr))))
        x_old = np.linspace(0.0, 1.0, num=src_len, endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=dst_len, endpoint=False)
        return np.interp(x_new, x_old, audio).astype(np.float32, copy=False)

    def _load_audio(self, audio_file, target_sampling_rate=None):
        # Decode WAV directly so transformers does not require ffmpeg for filename inputs.
        audio, sampling_rate = sf.read(audio_file, dtype="float32", always_2d=False)

        if isinstance(audio, np.ndarray) and audio.ndim > 1:
            # Convert multi-channel audio to mono.
            audio = np.mean(audio, axis=1)

        if not isinstance(audio, np.ndarray):
            audio = np.asarray(audio, dtype=np.float32)

        sampling_rate = int(sampling_rate)
        if target_sampling_rate is not None:
            target_sampling_rate = int(target_sampling_rate)
            if target_sampling_rate > 0 and sampling_rate != target_sampling_rate:
                audio = self._resample_audio(audio, sampling_rate, target_sampling_rate)
                sampling_rate = target_sampling_rate

        return audio.astype(np.float32, copy=False), int(sampling_rate)

    def _run_pipe(self, audio_array, sampling_rate, call_kwargs, legacy_parity=False):
        if not legacy_parity:
            return self.pipe({"raw": audio_array, "sampling_rate": sampling_rate}, **call_kwargs)

        # Legacy LiveCaptions uses more expensive decoding behavior.
        legacy_kwargs = dict(call_kwargs)
        gen_kwargs = dict(legacy_kwargs.get("generate_kwargs", {}))
        gen_kwargs.setdefault("num_beams", 5)
        gen_kwargs.setdefault("condition_on_prev_tokens", True)
        legacy_kwargs["generate_kwargs"] = gen_kwargs

        try:
            return self.pipe({"raw": audio_array, "sampling_rate": sampling_rate}, **legacy_kwargs)
        except Exception as exc:
            logging.warning(
                "Legacy-parity decode kwargs were not accepted (%s). Falling back to default decode kwargs.",
                exc,
            )
            return self.pipe({"raw": audio_array, "sampling_rate": sampling_rate}, **call_kwargs)

    def transcribe(self, audio_file, language="en", task="transcribe", chunk_length_s=30.0, batch_size=8):
        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"Audio file does not exist: {audio_file}")

        target_sr = int(getattr(self.processor.feature_extractor, "sampling_rate", 16000))
        audio_array, sampling_rate = self._load_audio(audio_file, target_sampling_rate=target_sr)

        generation_kwargs = {}
        if language:
            generation_kwargs["language"] = language
        if task:
            generation_kwargs["task"] = task

        call_kwargs = {
            "chunk_length_s": chunk_length_s,
            "batch_size": batch_size,
            "return_timestamps": True,
        }
        if generation_kwargs:
            call_kwargs["generate_kwargs"] = generation_kwargs

        start_time = time.time()
        result = self.pipe({"raw": audio_array, "sampling_rate": sampling_rate}, **call_kwargs)
        end_time = time.time()

        text = result.get("text") if isinstance(result, dict) else str(result)
        segments = result.get("chunks") if isinstance(result, dict) else None

        return {
            "status": "live_captions_complete",
            "text": text,
            "segments": segments,
            "audio_duration": self._safe_audio_duration(audio_file),
            "processing_time": end_time - start_time,
            "audio_file": audio_file,
        }

    def transcribe_realtime(
        self,
        audio_file,
        language="en",
        task="transcribe",
        chunk_seconds=2.0,
        simulate_realtime=True,
        legacy_parity=True,
    ):
        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"Audio file does not exist: {audio_file}")

        target_sr = int(getattr(self.processor.feature_extractor, "sampling_rate", 16000))
        audio_array, sampling_rate = self._load_audio(audio_file, target_sampling_rate=target_sr)
        if chunk_seconds <= 0:
            raise ValueError("chunk_seconds must be > 0")

        chunk_samples = int(sampling_rate * chunk_seconds)
        if chunk_samples <= 0:
            raise ValueError("Invalid chunk size derived from sampling rate and chunk_seconds")

        generation_kwargs = {}
        if language:
            generation_kwargs["language"] = language
        if task:
            generation_kwargs["task"] = task

        start_wall = time.time()
        chunk_results = []
        all_text_parts = []
        accumulated_audio = np.array([], dtype=np.float32)

        total_samples = len(audio_array)
        chunk_count = (total_samples + chunk_samples - 1) // chunk_samples

        for idx in range(chunk_count):
            begin = idx * chunk_samples
            end = min((idx + 1) * chunk_samples, total_samples)
            raw_chunk = audio_array[begin:end]

            if raw_chunk.size < chunk_samples:
                pad_width = chunk_samples - raw_chunk.size
                raw_chunk = np.pad(raw_chunk, (0, pad_width), mode="constant", constant_values=0.0)

            chunk_send_time = time.time()
            call_kwargs = {"return_timestamps": True}
            if generation_kwargs:
                call_kwargs["generate_kwargs"] = generation_kwargs

            if legacy_parity:
                accumulated_audio = np.concatenate((accumulated_audio, raw_chunk))
                inference_audio = accumulated_audio
            else:
                inference_audio = raw_chunk

            model_start_time = time.time()
            pipe_result = self._run_pipe(
                audio_array=inference_audio,
                sampling_rate=sampling_rate,
                call_kwargs=call_kwargs,
                legacy_parity=legacy_parity,
            )
            model_end_time = time.time()
            chunk_recv_time = time.time()

            chunk_text = pipe_result.get("text", "") if isinstance(pipe_result, dict) else str(pipe_result)
            if chunk_text:
                all_text_parts.append(chunk_text.strip())

            chunk_payload = {
                "chunk_index": idx,
                "chunk_start_sec": begin / float(sampling_rate),
                "chunk_end_sec": min((idx + 1) * chunk_seconds, total_samples / float(sampling_rate)),
                "processing_time": chunk_recv_time - chunk_send_time,
                "model_processing_time": model_end_time - model_start_time,
                "sent_at": chunk_send_time,
                "received_at": chunk_recv_time,
                "inference_audio_seconds": len(inference_audio) / float(sampling_rate),
                "text": chunk_text,
                "segments": pipe_result.get("chunks") if isinstance(pipe_result, dict) else None,
            }
            chunk_results.append(chunk_payload)

            if simulate_realtime and idx < chunk_count - 1:
                elapsed = time.time() - chunk_send_time
                remaining = chunk_seconds - elapsed
                if remaining > 0:
                    time.sleep(remaining)

        end_wall = time.time()

        return {
            "status": "live_captions_complete",
            "text": " ".join([t for t in all_text_parts if t]),
            "chunks": chunk_results,
            "audio_duration": total_samples / float(sampling_rate),
            "processing_time": end_wall - start_wall,
            "audio_file": audio_file,
            "chunk_seconds": chunk_seconds,
            "simulate_realtime": simulate_realtime,
            "legacy_parity": bool(legacy_parity),
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
                self._write_json(200, {"status": "ok"})
            else:
                self._write_json(404, {"error": "not found"})

        def do_POST(self):
            if self.path not in {"/transcribe", "/transcribe_realtime"}:
                self._write_json(404, {"error": "not found"})
                return

            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw_body = self.rfile.read(length) if length > 0 else b"{}"
                payload = json.loads(raw_body.decode("utf-8"))

                audio_file = payload.get("audio_file", "")
                if not audio_file:
                    self._write_json(400, {"error": "audio_file is required"})
                    return

                if self.path == "/transcribe_realtime":
                    simulate_realtime = payload.get("simulate_realtime", True)
                    if isinstance(simulate_realtime, str):
                        simulate_realtime = simulate_realtime.lower() in {"1", "true", "yes", "on"}

                    legacy_parity = payload.get("legacy_parity", True)
                    if isinstance(legacy_parity, str):
                        legacy_parity = legacy_parity.lower() in {"1", "true", "yes", "on"}

                    response = app.transcribe_realtime(
                        audio_file=audio_file,
                        language=payload.get("language", "en"),
                        task=payload.get("task", "transcribe"),
                        chunk_seconds=float(payload.get("chunk_seconds", 2.0)),
                        simulate_realtime=simulate_realtime,
                        legacy_parity=legacy_parity,
                    )
                    logging.info(
                        "Realtime transcription complete file=%s processing_time=%.3fs chunk_seconds=%.2f legacy_parity=%s",
                        audio_file,
                        float(response.get("processing_time", 0.0)),
                        float(payload.get("chunk_seconds", 2.0)),
                        bool(legacy_parity),
                    )
                else:
                    response = app.transcribe(
                        audio_file=audio_file,
                        language=payload.get("language", "en"),
                        task=payload.get("task", "transcribe"),
                        chunk_length_s=float(payload.get("chunk_length_s", 30.0)),
                        batch_size=int(payload.get("batch_size", 8)),
                    )
                    logging.info(
                        "Transcription complete file=%s processing_time=%.3fs",
                        audio_file,
                        float(response.get("processing_time", 0.0)),
                    )
                self._write_json(200, response)
            except Exception as e:
                logging.exception("Transcription request failed")
                self._write_json(500, {"error": str(e)})

        def log_message(self, format, *args):
            return

    return Handler


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="[livecaptionshf_server] %(asctime)s %(levelname)s %(message)s",
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

    app = LiveCaptionsHFServer(
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

