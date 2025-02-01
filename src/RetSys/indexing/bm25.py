import nltk
import numpy as np
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from typing import List, Tuple, Any
from .kv_store import KVStore
from .kv_store import TextType


class BM25(KVStore):
    def __init__(self, index_name: str):
        """
        Initialize the BM25 class.

        :param index_name: The name of the index.
        :type index_name: str
        """
        super().__init__(index_name, "bm25")

        nltk.download("punkt")
        nltk.download("stopwords")

        self._tokenizer = nltk.word_tokenize
        self._stop_words = set(nltk.corpus.stopwords.words("english"))
        self._stemmer = nltk.stem.PorterStemmer().stem
        self.index = None  # BM25 index

    def _encode_batch(
        self, texts: str, type: TextType, show_progress_bar: bool = True
    ) -> List[str]:
        """
        Encode a batch of texts.

        :param texts: The texts to encode.
        :type texts: str
        :param type: The type of text.
        :type type: TextType
        :param show_progress_bar: Whether to show a progress bar.
        :type show_progress_bar: bool
        :return: The encoded texts.
        :rtype: List[str]
        """
        # lowercase, tokenize, remove stopwords, and stem
        tokens_list = []
        for text in tqdm(texts, disable=not show_progress_bar):
            tokens = self._tokenizer(text.lower())
            tokens = [token for token in tokens if token not in self._stop_words]
            tokens = [self._stemmer(token) for token in tokens]
            tokens_list.append(tokens)
        return tokens_list

    def _query(self, encoded_query: List[str], n: int) -> List[int]:
        """
        Query the index.

        :param encoded_query: The encoded query.
        :type encoded_query: List[str]
        :param n: The number of results to return.
        :type n: int
        :return: The indices of the results.
        :rtype: List[int]
        """
        top_indices = np.argsort(self.index.get_scores(encoded_query))[::-1][
            :n
        ].tolist()
        return top_indices

    def clear(self) -> None:
        """
        Clear the index.
        """
        super().clear()
        self.index = None

    def create_index(self, key_value_pairs: List[Tuple[str, Any]]) -> None:
        """
        Create the index.

        :param key_value_pairs: The key-value pairs to create the index from.
        :type key_value_pairs: List[Tuple[str, Any]]
        """
        super().create_index(key_value_pairs)
        self.index = BM25Okapi(self.encoded_keys)

    def load(self, dir_name: str) -> None:
        """
        Load the index from disk.

        :param dir_name: The directory to load the index from.
        :type dir_name: str
        """
        super().load(dir_name)
        self._tokenizer = nltk.word_tokenize
        self._stop_words = set(nltk.corpus.stopwords.words("english"))
        self._stemmer = nltk.stem.PorterStemmer().stem
        return self
