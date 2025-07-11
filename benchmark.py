import os
import time
import memorag


def run(baseline: bool):
    os.environ["MEMORAG_BASELINE"] = "1" if baseline else "0"
    start = time.perf_counter()
    index, chunk_map = memorag.ingest_documents("sample_docs")
    ingest = time.perf_counter() - start

    query = "How can I teach history to my student that have ADHD? What are the best pratices"

    start = time.perf_counter()
    clue = memorag.generate_clue(query, index, chunk_map)
    clue_t = time.perf_counter() - start

    start = time.perf_counter()
    chunks = memorag.retrieve_chunks(clue, index, chunk_map)
    retrieval = time.perf_counter() - start

    start = time.perf_counter()
    answer = memorag.generate_final_answer(query, chunks)
    answer_t = time.perf_counter() - start

    return {
        "ingest_time": ingest,
        "clue_time": clue_t,
        "retrieval_time": retrieval,
        "answer_time": answer_t,
        "api_calls": dict(memorag.metrics),
    }


def main():
    memorag.load_env()
    print("Baseline run")
    base = run(True)
    print(base)
    print("Optimized run")
    opt = run(False)
    print(opt)


if __name__ == "__main__":
    main()
