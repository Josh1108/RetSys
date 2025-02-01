from typing import List
from .build_datasets import DatasetConverter
from .build_index import IndexBuilder
import datasets
import os

# Two workflows:
# 1. Create new index:
#    ret = Retriever(index_type, index_name, index_save_dir)
#    ret.load_data(...) # Converts data and builds index
# 2. Load existing index:
#    ret = Retriever.load_from_path(index_save_dir)

class Retriever:
    def __init__(self, index_type: str, index_name: str, index_save_dir: str):
        """
        Initialize the Retriever class.

        :param index_type: The type of index to build.
        :type index_type: str
        :param index_name: The name of the index.
        :type index_name: str
        :param index_save_dir: The directory to save the index.
        :type index_save_dir: str
        """
        self.index_type = index_type
        self.index_name = index_name 
        self.save_dir = index_save_dir
        self.index = None

    @classmethod
    def load_from_path(cls, index_path: str):
        """
        Load an existing index from disk.

        :param index_path: The path to the index.
        :type index_path: str
        """
        index_dir = os.path.dirname(index_path)
        index_name = os.path.basename(index_path)
        index_type = index_name.split(".")[-1]
        retriever = cls(index_type, index_name, index_dir)


        # Load index details from save_dir

        index_builder = IndexBuilder(index_type=index_type, index_name=index_name, save_dir=index_dir)
        retriever.index = index_builder.load_index(index_path)
        print(f"Loaded index {index_name} from {index_path}")
        return retriever
    def insert_data_and_save_index(self, dir_path: str, dataset_name: str, private: bool = False,
                 save_locally: bool = False, save_on_hf_hub: bool = False, dataset_dir: str = ".", granularity: str = "paragraphs"):
        """
        Convert data and build new index.

        :param dir_path: The path to the directory containing data.
        :type dir_path: str
        :param dataset_name: The name of the dataset.
        :type dataset_name: str
        :param private: Whether the dataset should be private.
        :type private: bool
        :param save_locally: Whether to save the dataset locally.
        :type save_locally: bool
        :param save_on_hf_hub: Whether to save the dataset on the Hugging Face Hub.
        :type save_on_hf_hub: bool
        :param dataset_dir: The directory to save the dataset.
        :type dataset_dir: str
        :param granularity: The granularity of the index.
        :type granularity: str
        """
        # Convert raw data to dataset
        dataset_converter = DatasetConverter()
        dataset_converter.run(dir_path, dataset_name, private, save_locally, save_on_hf_hub, dataset_dir)
        
        # Load the converted dataset
        if save_locally:
            corpus_data = datasets.load_from_disk(os.path.join(dataset_dir, dataset_name))
        else:
            corpus_data = datasets.load_dataset(dataset_name, split="full")

        # Build and save the index
        index_builder = IndexBuilder(index_type=self.index_type, index_name=self.index_name, save_dir=self.save_dir, granularity=granularity)
        kv_pairs = index_builder.create_kv_pairs(corpus_data)
        index_builder.index.create_index(kv_pairs)
        index_builder.index.save(self.save_dir)
        self.index = index_builder.index

    def query(self, query: str, top_k: int = 10, return_keys: bool = False):
        """
        Query the index.

        :param query: The query to search for.
        :type query: str
        :param top_k: The number of results to return.
        :type top_k: int
        :param return_keys: Whether to return the keys.
        :type return_keys: bool
        """
        if self.index is None:
            raise ValueError("No index loaded. Either load_data() or load_from_path() must be called first")
        
        return self.index.query(query, top_k, return_keys)