import json
import logging

import faiss

from memorag import (
    load_env,
    generate_clue,
    retrieve_chunks,
    generate_final_answer,
)


def load_index(index_path: str = "memory.index", map_path: str = "memory_map.json"):
    index = faiss.read_index(index_path)
    with open(map_path, "r") as f:
        chunk_map = json.load(f)
    # keys should be int not str
    chunk_map = {int(k): v for k, v in chunk_map.items()}
    return index, chunk_map


def answer_query(query: str, index, chunk_map):
    clue = generate_clue(query)
    chunks = retrieve_chunks(clue, index, chunk_map)
    if not chunks:
        return "No relevant information found."
    return generate_final_answer(query, chunks)


def main():
    load_env()
    try:
        index, chunk_map = load_index()
    except Exception as err:
        logging.error("Failed to load index: %s", err)
        return

    print("Type 'exit' to quit.")
    while True:
        query = input("Query: ").strip()
        if not query or query.lower() == "exit":
            break
        print("Generating answer...\n")
        try:
            answer = answer_query(query, index, chunk_map)
            print(answer)
        except Exception as exc:
            logging.exception("Error while generating answer: %s", exc)
            print("An error occurred. Please check logs.")


if __name__ == "__main__":
    import dotenv
    dotenv.load_dotenv()
    logging.basicConfig(level=logging.INFO)
    main()
