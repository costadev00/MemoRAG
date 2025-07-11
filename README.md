# MemoRAG

**Full implementation from scratch of the paper “Boosting Long-Context Processing with Global Memory-Enhanced Retrieval & Argumentation”**

MemoRAG is a next‐generation Retrieval-Augmented Generation framework designed to empower LLMs to handle million-token inputs efficiently and accurately by building a compact global memory and retrieving only the most relevant evidence.

---

## 🔍 Key Features

- **Dual-System Architecture**  
  1. **Light Memory Module** compresses vast contexts into a compact KV cache  
  2. **Expressive Generator** uses memory-driven “clues” to fetch precise passages and craft high-quality answers  

- **State-of-the-Art Performance**  
  Outperforms standard RAG and advanced pipelines (HyDE, GraphRAG, RQ-RAG) on 20+ long-context benchmarks  

- **Memory Compression**  
  Reduces GPU memory footprint by up to 64× through configurable “memory tokens”  

- **Reinforcement Learning (RLGF)**  
  Finetunes the memory-to-generator loop for end-to-end accuracy gains  

- **Plug-&-Play**  
  Swap in your favorite retrievers (e.g., BGE-M3) and generators (e.g., Phi-3-mini-128K, GPT-4o)  

---

## 🚀 Quickstart

1. **Clone the repo**  
   ```bash
   git clone https://github.com/costadev00/MemoRAG.git
   cd MemoRAG
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the demo**
   ```bash
   python memorag.py
   ```

### Approach Overview

1. **Ingest documents** – `ingest_documents()` reads PDFs from `sample_docs/`, splits them into ~4096 token chunks, compresses each with a small LLM and stores embeddings in a FAISS HNSW index.
2. **Generate a clue** – `generate_clue(query, index, chunk_map)` uses the global memory to draft a short answer that guides retrieval.
3. **Retrieve relevant memory** – `retrieve_chunks()` expands the clue into a retrieval query, embeds it, and searches the index for matching chunks.
4. **Generate the final answer** – `generate_final_answer(query, chunks)` reads the referenced pages and crafts the final response.
5. The optional script `ingestor.py` writes the index to `memory.index` and `memory_map.json` for later reuse.

Set `MEMORAG_BASELINE=1` to disable caching and threading for baseline benchmarking (see `benchmark.py`).


Run `benchmark.py` to compare baseline and optimized modes.

### Performance Notes

- Parallel ingestion and cached embeddings follow the memory construction algorithm (Sec.3.1).
- Clue generation and retrieval timings verify the dual-LLM split (Fig.3).

