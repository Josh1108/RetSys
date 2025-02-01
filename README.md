## Retrieval System

## Installation

```
pip3 install git+https://github.com/Josh1108/RetSys.git   
```

## Quick Start

```Python
from RetSys.indexing import Retriever

retriever = Retriever.load_from_path("retrieval_indices/maritime_docs.bm25")

print(retriever.query("How are the boundaries of the source water area determined", return_keys=True))
```
## Creating an index
```Python
from RetSys.indexing import Retriever

retriever = Retriever(index_type="bm25", index_name="maritime_docs", index_save_dir="retrieval_indices")
retriever.insert_data_and_save_index("folder_with_docs", "dataset_name", save_locally=True, granularity="paragraphs")
retriever.query("How are the boundaries of the source water area determined", return_keys=True)
```


