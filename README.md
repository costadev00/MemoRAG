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
