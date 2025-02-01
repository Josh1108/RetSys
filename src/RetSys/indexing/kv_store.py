import os
import pickle
from tqdm import tqdm
from enum import Enum
from typing import List, Tuple, Any

class TextType(Enum):
    KEY = 1
    QUERY = 2

class KVStore:
    """
    Base class for key-value stores.
    """
    def __init__(self, index_name: str, index_type: str) -> None:
        """
        Initialize the KVStore class.

        :param index_name: The name of the index.
        :type index_name: str
        :param index_type: The type of index.
        :type index_type: str
        """
        self.index_name = index_name
        self.index_type = index_type

        self.keys = []
        self.encoded_keys = []
        self.values = []

    def __len__(self) -> int:
        """
        Get the length of the index.

        :return: The length of the index.
        :rtype: int
        """
        return len(self.keys)

    def _encode(self, text: str, type: TextType) -> Any:
        """
        Encode a text.

        :param text: The text to encode.
        :type text: str
        :param type: The type of text.
        :type type: TextType
        :return: The encoded text.
        :rtype: Any
        """
        return self._encode_batch([text], type, show_progress_bar=False)[0]
    
    def _encode_batch(self, texts: List[str], type: TextType, show_progress_bar: bool = True) -> List[Any]: 
        """
        Encode a batch of texts.

        :param texts: The texts to encode.
        :type texts: List[str]
        :param type: The type of text.
        :type type: TextType
        :param show_progress_bar: Whether to show a progress bar.
        :type show_progress_bar: bool
        :return: The encoded texts.
        :rtype: List[Any]
        """
        raise NotImplementedError
    
    def _query(self, encoded_query: Any, n: int) -> List[int]:
        """
        Query the index.

        :param encoded_query: The encoded query.
        :type encoded_query: Any
        :param n: The number of results to return.
        :type n: int
        :return: The indices of the results.
        :rtype: List[int]
        """
        raise NotImplementedError
    
    def clear(self) -> None:
        """
        Clear the index.
        """
        self.keys = []
        self.encoded_keys = []
        self.values = []

    def create_index(self, key_value_pairs: List[Tuple[str, Any]]) -> None:
        """
        Create the index.

        :param key_value_pairs: The key-value pairs to create the index from.
        :type key_value_pairs: List[Tuple[str, Any]]
        """
        if len(self.keys) > 0:
            raise ValueError("Index is not empty. Please create a new index or clear the existing one.")
        
        for key, value in tqdm(key_value_pairs.items(), desc=f"Creating {self.index_name} index"):
            self.keys.append(key)
            self.values.append(value)
        self.encoded_keys = self._encode_batch(self.keys, TextType.KEY)

    def query(self, query_text: str, n: int, return_keys: bool = False) -> List[Any]:
        """
        Query the index.

        :param query_text: The query text.
        :type query_text: str
        :param n: The number of results to return.
        :type n: int
        :param return_keys: Whether to return the keys.
        :type return_keys: bool
        :return: The results.
        :rtype: List[Any]
        """
        encoded_query = self._encode(query_text, TextType.QUERY)
        indices = self._query(encoded_query, n)
        if return_keys:
            results = [(self.keys[i], self.values[i]) for i in indices]
        else:
            results = [self.values[i] for i in indices]
        return results

    def save(self, dir_name: str) -> None:
        """
        Save the index to disk.

        :param dir_name: The directory to save the index.
        :type dir_name: str
        """
        save_dict = {}
        for key, value in self.__dict__.items():
            if key[0] != "_":
                save_dict[key] = value

        print(f"Saving index to {os.path.join(dir_name, f'{self.index_name}.{self.index_type}')}")
        os.makedirs(dir_name, exist_ok=True)
        with open(os.path.join(dir_name, f"{self.index_name}.{self.index_type}"), 'wb') as file:
            pickle.dump(save_dict, file, protocol=pickle.HIGHEST_PROTOCOL)
        

    def load(self, file_path: str) -> None:
        """
        Load the index from disk.

        :param file_path: The file path to load the index from.
        :type file_path: str
        """
        if len(self.keys) > 0:
            raise ValueError("Index is not empty. Please create a new index or clear the existing one before loading from disk.")
        
        print(f"Loading index from {file_path}...")
        with open(file_path, 'rb') as file:
            pickle_data = pickle.load(file)
        
        for key, value in pickle_data.items():
            setattr(self, key, value)