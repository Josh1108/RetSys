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
retriever.query("What is the flowchart that functions as a decision tree for marine debris response to inform the response to debris in waterways and along shorelines after a disaster?", top_k = 2, return_page_number=True)
# Response:
[{'Page Number': '22', 'Location': ('/home/jkalra/directed_study/new_guides/US_Marine_Debris_Emergency_Response_Guide_2023-1.pdf_page_22', 1)}, {'Page Number': '23', 'Location': ('/home/jkalra/directed_study/new_guides/US_Marine_Debris_Emergency_Response_Guide_2023-1.pdf_page_23', 4)}]


