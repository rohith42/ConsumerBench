import argparse
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import faiss
import orjson
from sentence_transformers import SentenceTransformer


class LocalFaissRetriever:
    def __init__(
        self,
        index_path,
        docs_path,
        model_name,
        default_k=5,
        device="cpu",
        mps=100,
    ):
        self.device = str(device).lower()
        self.default_k = int(default_k)

        if self.device in {"gpu", "cuda", "cuda:0"} and int(mps) > 0:
            os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(int(mps))

        self.index = faiss.read_index(index_path)
        model_device = "cuda" if self.device in {"gpu", "cuda", "cuda:0"} else "cpu"
        self.model = SentenceTransformer(model_name, device=model_device)

        if self.device in {"gpu", "cuda", "cuda:0"}:
            try:
                if faiss.get_num_gpus() > 0:
                    gpu_res = faiss.StandardGpuResources()
                    self.index = faiss.index_cpu_to_gpu(gpu_res, 0, self.index)
                else:
                    print("No FAISS GPU detected; using CPU index.")
            except Exception as exc:
                print(f"FAISS GPU init failed; using CPU index. Error: {exc}")

        self.docs = {}
        with open(docs_path, "rb") as f:
            for line in f:
                obj = orjson.loads(line)
                self.docs[obj["doc_id"]] = obj["text"]

    def retrieve(self, query, k=None):
        top_k = int(k) if k is not None else self.default_k
        start_time = time.time()

        q = self.model.encode([query], normalize_embeddings=True).astype("float32")
        scores, ids = self.index.search(q, top_k)

        passages = []
        for doc_id, score in zip(ids[0].tolist(), scores[0].tolist()):
            if doc_id < 0:
                continue
            passages.append({
                "doc_id": int(doc_id),
                "score": float(score),
                "text": self.docs.get(doc_id, ""),
            })

        return {
            "status": "retrieval_complete",
            "query": query,
            "passages": passages,
            "total_time": time.time() - start_time,
        }


def make_handler(retriever):
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
            if self.path != "/retrieve":
                self._write_json(404, {"error": "not found"})
                return

            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw_body = self.rfile.read(length) if length > 0 else b"{}"
                payload = json.loads(raw_body.decode("utf-8"))

                query = payload.get("query", "")
                if not query:
                    self._write_json(400, {"error": "query is required"})
                    return

                result = retriever.retrieve(query=query, k=payload.get("k"))
                self._write_json(200, result)
            except Exception as e:
                self._write_json(500, {"error": str(e)})

        def log_message(self, format, *args):
            return

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--index-path", type=str, required=True)
    parser.add_argument("--docs-path", type=str, required=True)
    parser.add_argument("--model-name", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--default-k", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--mps", type=int, default=100)
    args = parser.parse_args()

    retriever = LocalFaissRetriever(
        index_path=args.index_path,
        docs_path=args.docs_path,
        model_name=args.model_name,
        default_k=args.default_k,
        device=args.device,
        mps=args.mps,
    )

    server = ThreadingHTTPServer((args.host, args.port), make_handler(retriever))
    server.serve_forever()


if __name__ == "__main__":
    main()
