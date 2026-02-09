import os
import sys
from typing import Any, Dict
import faiss
from sentence_transformers import SentenceTransformer
import orjson

repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(repo_dir)

from applications.application import Application

class LocalFaissRetriever:
    """
    Local retriever callable:
      passages = retriever(query, k=5)
    returns list[dict]: {"doc_id", "score", "text"}
    """
    def __init__(
        self,
        index_path: str,
        docs_path: str,
        model_name: str,
        default_k: int = 5,
        device: str = "cpu",
    ):
        self.index = faiss.read_index(index_path)
        self.device = device.lower()
        model_device = "cuda" if self.device in {"gpu", "cuda", "cuda:0"} else "cpu"
        self.model = SentenceTransformer(model_name, device=model_device)
        self.default_k = default_k

        if self.device in {"gpu", "cuda", "cuda:0"}:
            try:
                if faiss.get_num_gpus() > 0:
                    gpu_res = faiss.StandardGpuResources()
                    self.index = faiss.index_cpu_to_gpu(gpu_res, 0, self.index)
                else:
                    print("No FAISS GPU detected; using CPU index.")
            except Exception as exc:
                print(f"FAISS GPU init failed; using CPU index. Error: {exc}")

        # Simple docstore in memory. For huge corpora, switch to sqlite or mmap.
        self.docs = {}
        with open(docs_path, "rb") as f:
            for line in f:
                obj = orjson.loads(line)
                self.docs[obj["doc_id"]] = obj["text"]

    def __call__(self, query: str, k: int | None = None):
        k = k or self.default_k
        q = self.model.encode([query], normalize_embeddings=True).astype("float32")
        scores, ids = self.index.search(q, k)

        out = []
        for doc_id, score in zip(ids[0].tolist(), scores[0].tolist()):
            if doc_id < 0:
                continue
            out.append({"doc_id": doc_id, "score": float(score), "text": self.docs[doc_id]})
        return out

class Retriever(Application):
    def __init__(self):
        super().__init__()
        self.retriever = None

    def run_setup(self, *args, **kwargs):
        print("Retriever setup")
        # hard guard against accidental OpenAI usage
        os.environ.pop("OPENAI_API_KEY", None)
        os.environ.pop("OPENAI_API_BASE", None)

        index_path = kwargs.get("index_path", self.config["index_path"])
        docs_path = kwargs.get("docs_path", self.config["docs_path"])
        model_name = kwargs.get("model_name", self.config["model_name"])
        default_k = kwargs.get("default_k", self.config["default_k"])
        device = kwargs.get("device", self.config["device"])

        self.retriever = LocalFaissRetriever(
            index_path=index_path,
            docs_path=docs_path,
            model_name=model_name,
            default_k=default_k,
            device=device,
        )
        return {"status": "setup_complete", "config": self.config}

    def run_cleanup(self, *args, **kwargs):
        print("Retriever cleanup")
        self.retriever = None
        return {"status": "cleanup_complete"}

    def run_application(self, *args, **kwargs):
        print("Retriever application")
        if self.retriever is None:
            raise RuntimeError("Retriever is not set up.")

        query = kwargs.get("query", self.config["query"])
        k = kwargs.get("k", self.config["default_k"])
        passages = self.retriever(query, k=k)
        return {"status": "retrieval_complete", "query": query, "passages": passages}

    def get_default_config(self) -> Dict[str, Any]:
        return {
            "index_path": f"{repo_dir}/applications/Retriever/rag_index/corpus.faiss",
            "docs_path": f"{repo_dir}/applications/Retriever/rag_index/docs.jsonl",
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "default_k": 5,
            "device": "cpu",
            "query": "What is retrieval augmented generation?",
        }

    def load_dataset(self, *args, **kwargs):
        print("Retriever loading dataset")
        return {"status": "dataset_loaded"}