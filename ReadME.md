## Retrieval System

Datasets on HuggingFace:
- Jo1811/maritime_docs

## Creating an index

This script creates a retrieval index for text-based data using various indexing methods (e.g., BM25, Instructor, E5, GTR). The index can be used for information retrieval tasks like finding relevant documents, paragraphs, or propositions from a dataset based on a query.

Features
Supports multiple indexing methods:
- BM25: Classical term-frequency-based retrieval.
- Instructor: Embedding-based retrieval with customizable instructions.
- E5: Efficient embedding models.
- GTR: Google T5 retrieval embeddings.


Flexible key-value pair generation for indexing:
- paragraphs: Extracts paragraphs from documents for retrieval.
- propositions: Extracts propositions from documents for retrieval.
Saves the index to a specified directory for future use.

```
python create_index.py --index_type <index_type> --key <key> [--dataset_path <dataset_path>] [--index_root_dir <index_root_dir>]

```

## Running a Query

```
python query_index.py --index_name <index_name> [--index_root_dir <index_root_dir>] [--top_k <top_k>]
```


