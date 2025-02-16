import os
import argparse
import datasets
from typing import List
from . import utils
from .kv_store import KVStore


class IndexBuilder:
    def __init__(self, index_type: str, index_name: str, save_dir: str, granularity: str = "paragraphs"):
        """
        Initialize the IndexBuilder class.

        :param index_type: The type of index to build.
        :type index_type: str
        :param index_name: The name of the index.
        :type index_name: str
        :param save_dir: The directory to save the index.
        :type save_dir: str
        :param granularity: The granularity of the index.
        :type granularity: str, defaults to "paragraphs"
        """
        self.index_type = index_type
        self.index_name = index_name
        self.granularity = granularity
        self.save_dir = save_dir
        self.index = self.initialize_index()
    def initialize_index(self) -> KVStore:
        """
        Initialize the index.

        :raises ValueError: If the index type is not valid.
        :raises ValueError: If the granularity is not valid.
        :return: The index.
        :rtype: KVStore
        """
        if self.index_type == "bm25":
            from .bm25 import BM25

            index = BM25(self.index_name)
        elif self.index_type == "instructor":
            from .instructor import Instructor

            if self.granularity == "propositions":
                query_instruction = "Represent the question for retrieving propositions from relevant documents:"
                key_instruction = "Represent the propositions of a document for retrieval:"
            elif self.granularity == "paragraphs":
                query_instruction = "Represent the question for retrieving passages from relevant documents:"
                key_instruction = "Represent the passage from the documents for retrieval:"
            else:
                raise ValueError("Invalid granularity, must be 'propositions' or 'paragraphs'")
            index = Instructor(self.index_name, key_instruction, query_instruction)
        elif self.index_type == "e5":
            from .e5 import E5

            index = E5(self.index_name)
        elif self.index_type == "gtr":
            from .gtr import GTR

            index = GTR(self.index_name)
        else:
            raise ValueError("Invalid index type")
        return index
    
    def create_kv_pairs(self, data: List[dict]) -> dict:
        """
        Create key-value pairs for the index.

        :param data: The data to create key-value pairs from.
        :type data: List[dict]
        :return: The key-value pairs.
        :rtype: dict
        """
        if self.granularity == "propositions":
            kv_pairs = {}
            propositions = utils.get_clean_propositions(data)  # load libraries once
            for i, record in enumerate(data):
                corpusid = utils.get_clean_corpusid(record)
                for proposition_idx, proposition in enumerate(propositions[i]):
                    kv_pairs[proposition] = (corpusid, proposition_idx)
        elif self.granularity == "paragraphs":
            kv_pairs = {}
            for record in data:
                corpusid = utils.get_clean_corpusid(record)
                paragraphs = utils.get_clean_paragraphs(record)
                for paragraph_idx, paragraph in enumerate(paragraphs):
                    kv_pairs[paragraph] = (corpusid, paragraph_idx)
        
        return kv_pairs
    def load_index(self, index_path: str) -> KVStore:
        """
        Load an existing index from disk.

        :param index_path: The path to the index.
        :type index_path: str
        :raises ValueError: If the index type is not valid.
        :return: The index.
        :rtype: KVStore
        """
        index_type = os.path.basename(index_path).split(".")[-1]
        if index_type == "bm25":
            from .bm25 import BM25

            index = BM25(None).load(index_path)
        elif index_type == "instructor":
            from .instructor import Instructor

            index = Instructor(None, None, None).load(index_path)
        elif index_type == "e5":
            from .e5 import E5

            index = E5(None).load(index_path)
        elif index_type == "gtr":
            from .gtr import GTR

            index = GTR(None).load(index_path)
        else:
            raise ValueError("Invalid index type")
        return index

