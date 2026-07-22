import csv
import json
import time
import numpy as np
from pathlib import Path
import requests

API_KEY = "sk-jxuxhcvreesogxowbmrtrycgbbgrqeicmxypcjqerlcnduxn"
API_URL = "https://api.siliconflow.cn/v1/embeddings"
MODEL = "BAAI/bge-large-en-v1.5"
BATCH_SIZE = 12
MIN_CALL_GAP = 0.05
MAX_TEXT_CHARS = 1500

MEMORY_CSV = "D:/memorytable/cassandra-kg-memory/results/locomo_memory_records.csv"
QA_CSV = "D:/memorytable/cassandra-kg-memory/results/locomo_qa_records.csv"
OUT_DIR = Path("D:/memorytable/cassandra-kg-memory/results")

MEM_EMB_FILE = OUT_DIR / "locomo_memory_bge_large.npy"
QA_EMB_FILE = OUT_DIR / "locomo_qa_bge_large.npy"
MEM_IDS_FILE = OUT_DIR / "locomo_memory_ids_bge.txt"
QA_IDS_FILE = OUT_DIR / "locomo_qa_ids_bge.txt"


def load_memory_texts(path):
    ids = []
    texts = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            mid = row["memory_id"].strip()
            if not mid:
                continue
            txt = f"{row['speaker'].strip()}: {row['text'].strip()}"
            if len(txt) > MAX_TEXT_CHARS:
                txt = txt[:MAX_TEXT_CHARS]
            ids.append(mid)
            texts.append(txt)
    return ids, texts


def load_qa_texts(path):
    ids = []
    texts = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            qid = row["qa_id"].strip()
            if not qid:
                continue
            q = row["question"].strip()
            if len(q) > MAX_TEXT_CHARS:
                q = q[:MAX_TEXT_CHARS]
            ids.append(qid)
            texts.append(q)
    return ids, texts


def encode_batch(texts, is_query=False, retries=8):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    batch_texts = []
    for t in texts:
        if is_query:
            batch_texts.append(t)
        else:
            batch_texts.append(t)

    data = {
        "model": MODEL,
        "input": batch_texts,
        "encoding_format": "float",
    }

    last_error = None
    for attempt in range(retries):
        try:
            resp = requests.post(API_URL, json=data, headers=headers, timeout=90)
            if resp.status_code == 429:
                wait = min(2 ** (attempt + 1), 60)
                print(f"  rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            result = resp.json()
            data_list = sorted(result["data"], key=lambda x: x.get("index", 0))
            embeddings = [d["embedding"] for d in data_list]
            if len(embeddings) != len(batch_texts):
                print(f"  WARNING: got {len(embeddings)} embeds for {len(batch_texts)} texts")
                if len(embeddings) < len(batch_texts):
                    embeddings.extend([[0.0] * 1024] * (len(batch_texts) - len(embeddings)))
                else:
                    embeddings = embeddings[:len(batch_texts)]
            return embeddings
        except requests.exceptions.Timeout:
            wait = min(2 ** (attempt + 1), 30)
            print(f"  timeout (attempt {attempt+1}/{retries}), waiting {wait}s...")
            time.sleep(wait)
            last_error = "timeout"
        except requests.exceptions.ConnectionError as e:
            wait = min(2 ** (attempt + 1), 30)
            print(f"  connection error (attempt {attempt+1}/{retries}): {e}, waiting {wait}s...")
            time.sleep(wait)
            last_error = str(e)
        except Exception as e:
            wait = min(2 ** (attempt + 1), 20)
            print(f"  error (attempt {attempt+1}/{retries}): {e}, waiting {wait}s...")
            time.sleep(wait)
            last_error = str(e)

    raise RuntimeError(f"All {retries} retries failed. Last error: {last_error}")


def encode_with_progress(ids, texts, label, is_query=False):
    all_embeddings = []
    total = len(texts)
    print(f"Encoding {total} {label} (model={MODEL})...")
    print(f"  batch_size={BATCH_SIZE}, est calls={(total + BATCH_SIZE - 1) // BATCH_SIZE}")

    for batch_start in range(0, total, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total)
        batch_texts = texts[batch_start:batch_end]
        batch_ids = ids[batch_start:batch_end]

        embs = encode_batch(batch_texts, is_query=is_query)
        all_embeddings.extend(embs)

        done = batch_end
        pct = 100.0 * done / total
        print(f"  [{done}/{total}] {pct:.1f}%")

        if batch_end < total:
            time.sleep(MIN_CALL_GAP)

    print(f"  Done: {len(all_embeddings)} embeddings")
    return np.array(all_embeddings, dtype=np.float32)


def main():
    mem_ids, mem_texts = load_memory_texts(MEMORY_CSV)
    print(f"Loaded {len(mem_texts)} memories from {MEMORY_CSV}")

    qa_ids, qa_texts = load_qa_texts(QA_CSV)
    print(f"Loaded {len(qa_texts)} QA queries from {QA_CSV}")
    print()

    mem_arr = encode_with_progress(mem_ids, mem_texts, "memories", is_query=False)
    print(f"Memory embeddings: {mem_arr.shape}\n")

    qa_arr = encode_with_progress(qa_ids, qa_texts, "queries", is_query=True)
    print(f"Query embeddings: {qa_arr.shape}\n")

    np.save(MEM_EMB_FILE, mem_arr)
    np.save(QA_EMB_FILE, qa_arr)
    print(f"Saved {MEM_EMB_FILE}")

    with open(MEM_IDS_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(mem_ids))

    with open(QA_IDS_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(qa_ids))

    print(f"Saved {MEM_IDS_FILE}")

    print()
    print("All done. Files:")
    print(f"  {MEM_EMB_FILE}  ({mem_arr.shape})")
    print(f"  {QA_EMB_FILE}   ({qa_arr.shape})")
    print(f"  {MEM_IDS_FILE}")
    print(f"  {QA_IDS_FILE}")


if __name__ == "__main__":
    main()
