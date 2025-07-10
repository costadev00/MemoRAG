import os
from dotenv import load_dotenv
import openai
import faiss
import numpy as np
from pathlib import Path
from PyPDF2 import PdfReader
import logging

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

def ingest_documents(directory: str):
    """Ingest PDF documents and build a FAISS index of compressed embeddings."""

    embeddings = []
    chunk_map = {}
    chunk_id = 0

    for path in Path(directory).rglob('*.pdf'):
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

        for i in range(0, len(tokens), 4096):
            end = min(i + 4096, len(tokens))
            chunk_text = ' '.join(tokens[i:end])
            start_page = token_pages[i] if token_pages else 1
            end_page = token_pages[end - 1] if token_pages else start_page
            page_range = (
                f"{start_page}" if start_page == end_page else f"{start_page}-{end_page}"
            )

            try:
                resp = openai.chat.completions.create(
                    model="o4-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": "Compress this chunk into key-value memory.",
                        },
                        {"role": "user", "content": chunk_text},
                    ],
                )
                compressed = resp.choices[0].message.content

                emb_resp = openai.embeddings.create(
                    model="text-embedding-3-large",
                    input=compressed,
                )
                vector = np.array(emb_resp.data[0].embedding, dtype="float32")

                embeddings.append(vector)
                chunk_map[chunk_id] = {
                    "filename": path.name,
                    "pages": page_range,
                }
                chunk_id += 1
            except Exception as err:
                logging.warning(
                    f"Embedding generation failed for {path.name} pages {page_range}: {err}"
                )

    if not embeddings:
        raise ValueError("No documents ingested for indexing.")

    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.vstack(embeddings))
    return index, chunk_map


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


def retrieve_chunks(clue: str, index, chunk_map: dict[int, dict], k: int = 3) -> list[str]:
    """Retrieve top-k relevant chunks from the FAISS index using the clue."""
    try:
        resp = openai.chat.completions.create(
            model="o4-mini",
            messages=[
                {"role": "system", "content": "Extract key phrases for retrieval."},
                {"role": "user", "content": clue}
            ]
        )
        retrieval_query = resp.choices[0].message.content

        emb_resp = openai.embeddings.create(
            model="text-embedding-3-large",
            input=retrieval_query
        )
        query_vec = np.array(emb_resp.data[0].embedding, dtype='float32').reshape(1, -1)

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
    index, chunk_map = ingest_documents('sample_docs')

    query = "How can I teach history to my student that have ADHD? What are the best pratices"
    clue = generate_clue(query)
    relevant_chunks = retrieve_chunks(clue, index, chunk_map)
    if not relevant_chunks:
        print("No relevant chunks found.")
        return
    final_answer = generate_final_answer(query, relevant_chunks)
    print("Final answer:\n", final_answer)


if __name__ == '__main__':
    main()
