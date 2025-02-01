import sentence_transformers
import numpy as np
from typing import List, Any
from sklearn.metrics.pairwise import cosine_similarity
from .kv_store import KVStore
from .kv_store import TextType
from . import utils

class GTR(KVStore):
    """
    GTR index class.
    """
    def __init__(self, index_name: str, model_path: str = "sentence-transformers/gtr-t5-large"):
        super().__init__(index_name, 'gtr')
        self.model_path = model_path
        self._model = sentence_transformers.SentenceTransformer(model_path, device="cuda", cache_folder=utils.get_cache_dir())
    
    def _encode_batch(self, texts: List[str], type: TextType, show_progress_bar: bool = True) -> List[Any]:
        """
        Encode a batch of texts.

        :param texts: The texts to encode.
        :type texts: List[str]
        :param type: The type of text.
        :type type: TextType
        :param show_progress_bar: Whether to show a progress bar.
        :type show_progress_bar: bool, optional
        :return: The encoded texts.
        :rtype: List[Any]
        """
        return self._model.encode(texts, batch_size=256, show_progress_bar=show_progress_bar).astype(np.float16)
    
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
        cosine_similarities = cosine_similarity([encoded_query], self.encoded_keys)[0]
        top_indices = cosine_similarities.argsort()[-n:][::-1]
        return top_indices
    
    def load(self, path: str):
        """
        Load the index from disk.

        :param path: The path to load the index from.
        :type path: str
        :return: The index.
        :rtype: GTR
        """
        super().load(path)
        self._model = sentence_transformers.SentenceTransformer(self.model_path, device="cuda", cache_folder=utils.get_cache_dir())
        return self
        
