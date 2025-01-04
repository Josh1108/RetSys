import os
import argparse
import datasets
from tqdm import tqdm
from utils import utils
from indexing.kv_store import KVStore


def load_index(index_path: str) -> KVStore:
    index_type = os.path.basename(index_path).split(".")[-1]
    if index_type == "bm25":
        from indexing.bm25 import BM25

        index = BM25(None).load(index_path)
    elif index_type == "instructor":
        from indexing.instructor import Instructor

        index = Instructor(None, None, None).load(index_path)
    elif index_type == "e5":
        from indexing.e5 import E5

        index = E5(None).load(index_path)
    elif index_type == "gtr":
        from indexing.gtr import GTR

        index = GTR(None).load(index_path)
    elif index_type == "grit":
        from indexing.grit import GRIT

        index = GRIT(None, None).load(index_path)
    else:
        raise ValueError("Invalid index type")
    return index


parser = argparse.ArgumentParser()
parser.add_argument("--index_name", type=str, required=True)
parser.add_argument(
    "--index_root_dir", type=str, required=False, default="retrieval_indices"
)
parser.add_argument("--top_k", type=int, required=False, default=200)
args = parser.parse_args()

index = load_index(os.path.join(args.index_root_dir, args.index_name))

while True:
    query = input("Enter query: ")
    if query == "exit":
        break
    top_k = index.query(query, args.top_k, return_keys=True)
    print(top_k)
