import json
import logging
from pathlib import Path
import faiss
from memorag import ingest_documents


def main():
    logging.info("Starting ingestion process")
    try:
        index, chunk_map = ingest_documents("sample_docs/")
        faiss.write_index(index, "memory.index")
        with open("memory_map.json", "w") as f:
            json.dump(chunk_map, f)
        logging.info("Ingestion completed successfully")
    except Exception as exc:
        logging.exception("Ingestion failed: %s", exc)


if __name__ == "__main__":
    import dotenv
    dotenv.load_dotenv()
    logging.basicConfig(filename="ingestor.log", level=logging.INFO)
    main()
