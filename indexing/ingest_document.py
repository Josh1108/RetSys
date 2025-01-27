from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import os
from indexing.kv_store import KVStore
from utils import utils

app = FastAPI()

# Configuration
INDEX_ROOT_DIR = "retrieval_indices"
INDEX = None

class Document(BaseModel):
    corpus_id: str
    content: str

class IngestRequest(BaseModel):
    key: str  # "paragraphs" or "propositions"
    documents: List[Document]

class QueryRequest(BaseModel):
    query: str
    top_k: int = 10

def load_index(index_name: str) -> KVStore:
    index_path = os.path.join(INDEX_ROOT_DIR, index_name)
    if not os.path.exists(index_path):
        raise ValueError(f"Index {index_name} not found at {index_path}.")
    
    index_type = os.path.basename(index_path).split(".")[-1]
    if index_type == "bm25":
        from indexing.bm25 import BM25
        return BM25(None).load(index_path)
    elif index_type == "instructor":
        from indexing.instructor import Instructor
        return Instructor(None, None, None).load(index_path)
    elif index_type == "e5":
        from indexing.e5 import E5
        return E5(None).load(index_path)
    elif index_type == "gtr":
        from indexing.gtr import GTR
        return GTR(None).load(index_path)
    elif index_type == "grit":
        from indexing.grit import GRIT
        return GRIT(None, None).load(index_path)
    else:
        raise ValueError(f"Unsupported index type: {index_type}")

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/ingest/")
def ingest_documents(request: IngestRequest):
    global INDEX
    if not INDEX:
        raise HTTPException(status_code=500, detail="Index not loaded.")

    try:
        # Generate key-value pairs from new documents
        data = [{"corpus_id": doc.corpus_id, "content": doc.content} for doc in request.documents]
        kv_pairs = {}
        if request.key == "propositions":
            propositions = utils.get_clean_propositions(data)
            for i, record in enumerate(data):
                corpus_id = utils.get_clean_corpusid(record)
                for proposition_idx, proposition in enumerate(propositions[i]):
                    kv_pairs[proposition] = (corpus_id, proposition_idx)
        elif request.key == "paragraphs":
            for record in data:
                corpus_id = utils.get_clean_corpusid(record)
                paragraphs = utils.get_clean_paragraphs(record)
                for paragraph_idx, paragraph in enumerate(paragraphs):
                    kv_pairs[paragraph] = (corpus_id, paragraph_idx)
        else:
            raise ValueError("Invalid key")

        # Update the index
        INDEX.add_kv_pairs(kv_pairs)
        INDEX.save(INDEX_ROOT_DIR)
        return {"message": "Documents ingested successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def lifespan(app: FastAPI):
    global INDEX
    index_name = "maritime_docs.bm25"  # Replace with your actual index name
    try:
        INDEX = load_index(index_name)
        print("Index loaded successfully.")
    except Exception as e:
        print(f"Error loading index: {e}")
        INDEX = None
    yield  # Indicates startup is complete
    # Add any cleanup/shutdown logic here if needed
    print("Shutting down the API...")

app = FastAPI(lifespan=lifespan)

@app.post("/query/")
def query_index(query: str, top_k: int = 10):
    global INDEX
    if not INDEX:
        return {"error": "Index not loaded."}
    results = INDEX.query(query, top_k, return_keys=True)
    return {"results": results}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("indexing.ingest_document:app", host="127.0.0.1", port=8000, reload=True)
