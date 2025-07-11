import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

import faiss
import numpy as np
import openai
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import logging

USE_BASELINE = os.getenv("MEMORAG_BASELINE", "0") == "1"

metrics = defaultdict(float)

# ------------------------------------------------------------
# Environment setup
# ------------------------------------------------------------

def load_env(env_path: str = '.env') -> None:
    """Load environment variables from a .env file."""
    load_dotenv(env_path)
    openai.api_key = os.getenv('OPENAI_API_KEY')


def _time_call(key: str):
    """Context manager to measure execution time for metrics."""
    class _Timer:
        def __enter__(self):
            self.start = time.perf_counter()
            return self

        def __exit__(self, exc_type, exc, tb):
            metrics[key] += time.perf_counter() - self.start

    return _Timer()


@lru_cache(maxsize=1024)
def _cached_completion(messages: tuple):
    # Convert tuple of tuples back to list of dicts
    openai_messages = [{"role": role, "content": content} for role, content in messages]
    response = openai.chat.completions.create(model="o4-mini", messages=openai_messages)
    return response.choices[0].message.content


def _compress_chunk(text: str) -> str:
    metrics["completion_calls"] += 1
    if USE_BASELINE:
        response = openai.chat.completions.create(
            model="o4-mini",
            messages=[
                {"role": "system", "content": "Compress this chunk into key-value memory."},
                {"role": "user", "content": text},
            ],
        )
        return response.choices[0].message.content
    else:
        messages = (
            ("system", "Compress this chunk into key-value memory."),
            ("user", text),
        )
        return _cached_completion(messages)


@lru_cache(maxsize=1024)
def _cached_embedding(text: str) -> np.ndarray:
    resp = openai.embeddings.create(model="text-embedding-3-large", input=[text])
    vec = np.array(resp.data[0].embedding, dtype="float32")
    return vec


def _get_embedding(text: str) -> np.ndarray:
    metrics["embedding_calls"] += 1
    if USE_BASELINE:
        resp = openai.embeddings.create(model="text-embedding-3-large", input=text)
        return np.array(resp.data[0].embedding, dtype="float32")
    else:
        return _cached_embedding(text)

# ------------------------------------------------------------
# Ingest step
# ------------------------------------------------------------

def ingest_documents(directory: str):
    """Ingest PDF documents and build a FAISS index of compressed embeddings."""
    # Sec.3.1 - parallel memory construction

    embeddings: list[np.ndarray] = []
    chunk_map: dict[int, dict] = {}
    chunk_id = 0

    def process_chunk(chunk_text: str, pr: str):
        with _time_call("memory_time"):
            compressed = _compress_chunk(chunk_text)
            vector = _get_embedding(compressed)
        return vector, pr

    for path in Path(directory).rglob("*.pdf"):
        try:
            reader = PdfReader(str(path))
        except Exception as err:
            logging.warning(f"Failed to open {path.name}: {err}")
            continue

        tokens: list[str] = []
        token_pages: list[int] = []

        for p_idx, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception as err:
                logging.warning(f"Failed reading {path.name} page {p_idx}: {err}")
                text = ""

            words = text.split()
            tokens.extend(words)
            token_pages.extend([p_idx] * len(words))

        chunks = []
        page_ranges = []
        for i in range(0, len(tokens), 4096):
            end = min(i + 4096, len(tokens))
            chunk_text = " ".join(tokens[i:end])
            start_page = token_pages[i] if token_pages else 1
            end_page = token_pages[end - 1] if token_pages else start_page
            page_range = (
                f"{start_page}" if start_page == end_page else f"{start_page}-{end_page}"
            )
            chunks.append(chunk_text)
            page_ranges.append(page_range)

        results = []
        if USE_BASELINE:
            for c, pr in zip(chunks, page_ranges):
                results.append(process_chunk(c, pr))
        else:
            with ThreadPoolExecutor(max_workers=4) as ex:
                futures = [ex.submit(process_chunk, c, pr) for c, pr in zip(chunks, page_ranges)]
                for f in as_completed(futures):
                    results.append(f.result())

        for vector, pr in results:
            embeddings.append(vector)
            chunk_map[chunk_id] = {"filename": path.name, "pages": pr}
            chunk_id += 1

    if not embeddings:
        raise ValueError("No documents ingested for indexing.")

    dim = len(embeddings[0])
    # Sec. 4.1 recommends HNSW for scalable memory search
    index = faiss.IndexHNSWFlat(dim, 32)
    index.hnsw.efSearch = 64
    index.add(np.vstack(embeddings))
    return index, chunk_map


def _split_into_chunks(text: str, chunk_size: int = 4096) -> list[str]:
    """Split text into roughly ``chunk_size`` token chunks."""
    words = text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]


def build_memory_index(texts: list[str]):
    """Build a FAISS index storing compressed embeddings for each chunk."""
    # Sec.3.1 - text ingestion for global memory
    chunks: list[str] = []
    chunk_map: dict[int, str] = {}
    embeddings: list[np.ndarray] = []

    for text in texts:
        for chunk in _split_into_chunks(text):
            chunks.append(chunk)
            chunk_map[len(chunks) - 1] = chunk

    batch_size = 16
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        with _time_call("memory_time"):
            if USE_BASELINE:
                resp = openai.embeddings.create(model="text-embedding-3-large", input=batch)
                vecs = [np.array(d.embedding, dtype="float32") for d in resp.data]
            else:
                vecs = [_get_embedding(c) for c in batch]
        embeddings.extend(vecs)

    if not embeddings:
        raise ValueError("No documents ingested for indexing.")

    dim = len(embeddings[0])
    index = faiss.IndexHNSWFlat(dim, 32)
    index.hnsw.efSearch = 64
    index.add(np.vstack(embeddings))
    return index, chunk_map

# ------------------------------------------------------------
# Retrieval step
# ------------------------------------------------------------

def generate_clue(query: str) -> str:
    """Generate a short draft answer (the "clue") for the user query."""
    # Sec.3.2 - draft clue produced by expressive generator
    with _time_call("generator_time"):
        response = openai.chat.completions.create(
            model="o4-mini",
            messages=[{"role": "user", "content": query}]
        )
    return response.choices[0].message.content


def retrieve_chunks(clue: str, index, chunk_map: dict[int, dict], k: int = 3) -> list[str]:
    """Retrieve top-k relevant chunks from the FAISS index using the clue."""
    # Fig.3 - clue-driven retrieval from compact memory
    try:
        with _time_call("generator_time"):
            resp = openai.chat.completions.create(
                model="o4-mini",
                messages=[
                    {"role": "system", "content": "Extract key phrases for retrieval."},
                    {"role": "user", "content": clue}
                ]
            )
        retrieval_query = resp.choices[0].message.content

        query_vec = _get_embedding(retrieval_query).reshape(1, -1)

        distances, indices = index.search(query_vec, k)
        # Format each chunk as a string for joining later
        retrieved = [
            f"{chunk_map[idx]['filename']} (pages {chunk_map[idx]['pages']})"
            for idx in indices[0] if idx in chunk_map
        ]
        return retrieved
    except Exception as err:
        logging.warning(f"Chunk retrieval failed: {err}")
        return []


def generate_final_answer(query: str, retrieved_chunks: list[str]) -> str:
    """Generate the final answer using the retrieved chunks and original query."""
    # Sec.4.1 - generator refines clue with retrieved evidence
    context = "\n".join(retrieved_chunks)
    prompt = f"Context:\n{context}\n\nUser question: {query}"
    with _time_call("generator_time"):
        stream = openai.chat.completions.create(
            model="o4-mini",
            messages=[
                {"role": "system", "content": "Use the context to answer precisely."},
                {"role": "user", "content": prompt},
            ],
            stream=True,
        )
        parts = [s.choices[0].delta.content or "" for s in stream]
    return "".join(parts)

# ------------------------------------------------------------
# Example usage
# ------------------------------------------------------------

def main():
    load_env()
    index, chunk_map = ingest_documents('sample_docs')

    query = "How can I teach history to my student that have ADHD? What are the best pratices"
    clue = generate_clue(query)
    relevant_chunks = retrieve_chunks(clue, index, chunk_map)
    if not relevant_chunks:
        print("No relevant chunks found.")
        return
    final_answer = generate_final_answer(query, relevant_chunks)
    print("Final answer:\n", final_answer)
    print("Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.2f}")


if __name__ == '__main__':
    main()
