import os
import argparse
import datasets
from typing import List
from utils import utils
from indexing.kv_store import KVStore


def get_index_name(args: argparse.Namespace) -> str:
    return os.path.basename(args.dataset_path) + "." + args.key


def create_index(args: argparse.Namespace) -> KVStore:
    index_name = get_index_name(args)

    if args.index_type == "bm25":
        from indexing.bm25 import BM25

        index = BM25(index_name)
    elif args.index_type == "instructor":
        from indexing.instructor import Instructor

        if args.key == "propositions":
            query_instruction = "Represent the question for retrieving propositions from relevant documents:"
            key_instruction = "Represent the propositions of a document for retrieval:"
        elif args.key == "paragraphs":
            query_instruction = "Represent the question for retrieving passages from relevant documents:"
            key_instruction = "Represent the passage from the documents for retrieval:"
        else:
            raise ValueError("Invalid key")
        index = Instructor(index_name, key_instruction, query_instruction)
    elif args.index_type == "e5":
        from indexing.e5 import E5

        index = E5(index_name)
    elif args.index_type == "gtr":
        from indexing.gtr import GTR

        index = GTR(index_name)
    else:
        raise ValueError("Invalid index type")
    return index


def create_kv_pairs(data: List[dict], key: str) -> dict:
    if key == "propositions":
        kv_pairs = {}
        propositions = utils.get_clean_propositions(data)  # load libraries once
        for i, record in enumerate(data):
            corpusid = utils.get_clean_corpusid(record)
            for proposition_idx, proposition in enumerate(propositions[i]):
                kv_pairs[proposition] = (corpusid, proposition_idx)
    elif key == "paragraphs":
        kv_pairs = {}
        for record in data:
            corpusid = utils.get_clean_corpusid(record)
            paragraphs = utils.get_clean_paragraphs(record)
            for paragraph_idx, paragraph in enumerate(paragraphs):
                kv_pairs[paragraph] = (corpusid, paragraph_idx)
    else:
        raise ValueError("Invalid key")
    return kv_pairs


parser = argparse.ArgumentParser()
parser.add_argument(
    "--index_type", required=True
)  # bm25, instructor, dpr, e5, gtr, grit
parser.add_argument("--key", required=True)  # paragraphs, propositions
parser.add_argument("--dataset_path", required=False, default="Jo1811/maritime_docs")
parser.add_argument("--index_root_dir", required=False, default="retrieval_indices")
args = parser.parse_args()

corpus_data = datasets.load_dataset(args.dataset_path, split="full")
index = create_index(args)
kv_pairs = create_kv_pairs(corpus_data, args.key)
index.create_index(kv_pairs)

index_name = get_index_name(args)
index.save(args.index_root_dir)
