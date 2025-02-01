## Retrieval System

## Installation

```
pip install retrieval_lib
```

## Quick Start

```
from retrieval_lib.indexing import Retriever

retriever = Retriever.load_from_path("retrieval_indices/maritime_docs.bm25")

print(retriever.query("How are the boundaries of the source water area determined", return_keys=True))
```
## Creating an index
```
from retrieval_lib.indexing import Retriever

retriever = Retriever(index_type="bm25", index_name="maritime_docs", save_dir="retrieval_indices")
retriever.insert_data_and_save_index("folder_with_docs", "dataset_name", save_locally=True, granularity="paragraphs")
retriever.query("How are the boundaries of the source water area determined", return_keys=True)
```

