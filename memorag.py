import os
from dotenv import load_dotenv
import openai
import faiss
import numpy as np
from pathlib import Path

# ------------------------------------------------------------
# Environment setup
# ------------------------------------------------------------

def load_env(env_path: str = '.env') -> None:
    """Load environment variables from a .env file."""
    load_dotenv(env_path)
    openai.api_key = os.getenv('OPENAI_API_KEY')

# ------------------------------------------------------------
# Ingest step
# ------------------------------------------------------------

def ingest_documents(directory: str) -> list[str]:
    """Read all text documents from the given directory."""
    texts = []
    for path in Path(directory).glob('**/*'):
        if path.is_file():
            with open(path, 'r', encoding='utf-8') as f:
                texts.append(f.read())
    return texts


def _split_into_chunks(text: str, chunk_size: int = 4096) -> list[str]:
    """Split text into roughly ``chunk_size`` token chunks."""
    words = text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]


def build_memory_index(texts: list[str]):
    """Build a FAISS index storing compressed embeddings for each chunk."""
    chunks = []
    chunk_map = {}
    embeddings = []

    for doc_id, text in enumerate(texts):
        for chunk in _split_into_chunks(text):
            # Obtain embedding via OpenAI embeddings endpoint
            response = openai.embeddings.create(
                model="text-embedding-3-large",
                input=chunk
            )
            vector = np.array(response.data[0].embedding, dtype='float32')
            embeddings.append(vector)
            chunk_map[len(chunks)] = chunk
            chunks.append(chunk)

    if not embeddings:
        raise ValueError("No documents ingested for indexing.")

    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.vstack(embeddings))
    return index, chunk_map

# ------------------------------------------------------------
# Retrieval step
# ------------------------------------------------------------

def generate_clue(query: str) -> str:
    """Generate a short draft answer (the "clue") for the user query."""
    response = openai.chat.completions.create(
        model="o4-mini",
        messages=[{"role": "user", "content": query}]
    )
    return response.choices[0].message.content


def retrieve_chunks(clue: str, index, chunk_map: dict[int, str], k: int = 3) -> list[str]:
    """Retrieve top-k relevant chunks from the FAISS index using the clue."""
    # Extract salient phrases to form a retrieval query
    resp = openai.chat.completions.create(
        model="o4-mini",
        messages=[
            {"role": "system", "content": "Extract key phrases for retrieval."},
            {"role": "user", "content": clue}
        ]
    )
    retrieval_query = resp.choices[0].message.content

    # Correct: Use the embeddings endpoint for the retrieval query
    emb_resp = openai.embeddings.create(
        model="text-embedding-3-large",
        input=retrieval_query
    )
    query_vec = np.array(emb_resp.data[0].embedding, dtype='float32').reshape(1, -1)

    distances, indices = index.search(query_vec, k)
    retrieved = [chunk_map[idx] for idx in indices[0] if idx in chunk_map]
    return retrieved


def generate_final_answer(query: str, retrieved_chunks: list[str]) -> str:
    """Generate the final answer using the retrieved chunks and original query."""
    context = '\n'.join(retrieved_chunks)
    prompt = f"Context:\n{context}\n\nUser question: {query}"
    response = openai.chat.completions.create(
        model="o4-mini",
        messages=[{"role": "system", "content": "Use the context to answer precisely."},
                  {"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# ------------------------------------------------------------
# Example usage
# ------------------------------------------------------------

def main():
    load_env()
    texts = ingest_documents('sample_docs')
    index, chunk_map = build_memory_index(texts)

    query = "What is MemoRAG?"
    clue = generate_clue(query)
    relevant_chunks = retrieve_chunks(clue, index, chunk_map)
    final_answer = generate_final_answer(query, relevant_chunks)
    print("Final answer:\n", final_answer)


if __name__ == '__main__':
    main()
