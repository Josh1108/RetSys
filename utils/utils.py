import os
import json
from typing import List, Any, Tuple
from datasets import Dataset
from utils.openai_utils import OPENAIBaseEngine
import spacy
from spacy.pipeline import Sentencizer
from tqdm import tqdm

nlp = spacy.blank("en")
nlp.add_pipe("sentencizer")
nlp.max_length = 100000000
##### file reading and writing #####


def read_json(filename: str, silent: bool = False) -> List[Any]:
    with open(filename, "r") as file:
        if filename.endswith(".json"):
            data = json.load(file)
        elif filename.endswith(".jsonl"):
            data = [json.loads(line) for line in file]
        else:
            raise ValueError("Input file must be either a .json or .jsonl file")

    if not silent:
        print(f"Loaded {len(data)} records from {filename}")
    return data


def write_json(data: List[Any], filename: str, silent: bool = False) -> None:
    if filename.endswith(".json"):
        with open(filename, "w") as file:
            json.dump(data, file, indent=4)
    elif filename.endswith(".jsonl"):
        with open(filename, "w") as file:
            for item in data:
                file.write(json.dumps(item) + "\n")
    else:
        raise ValueError("Output file must be either a .json or .jsonl file")

    if not silent:
        print(f"Saved {len(data)} records to {filename}")


def read_txt(filename: str) -> str:
    with open(filename, "r") as file:
        text = file.read()
    return text


##### evaluation metrics #####


def calculate_recall(retrieved: List[int], relevant_docs: List[int]) -> float:
    num_relevant_retrieved = len(set(retrieved).intersection(set(relevant_docs)))
    num_relevant = len(relevant_docs)
    recall = num_relevant_retrieved / num_relevant if num_relevant > 0 else 0
    return recall


def calculate_ndcg(retrieved: List[int], relevant_docs: List[int]) -> float:
    dcg = 0
    for idx, docid in enumerate(retrieved):
        if docid in relevant_docs:
            dcg += 1 / (idx + 1)
    idcg = sum([1 / (idx + 1) for idx in range(len(relevant_docs))])
    ndcg = dcg / idcg if idcg > 0 else 0
    return ndcg


def calculate_ngram_overlap(query: str, text: str) -> float:
    query_ngrams = set(query.split())
    text_ngrams = set(text.split())
    overlap = len(query_ngrams.intersection(text_ngrams)) / len(query_ngrams)
    return overlap


##### reading fields from corpus_clean #####


def get_clean_corpusid(item: dict) -> str:
    return item["file_name"]


def get_clean_title(item: dict) -> str:
    return item["title"]


def get_clean_abstract(item: dict) -> str:
    return item["abstract"]


def get_clean_title_abstract(item: dict) -> str:
    title = get_clean_title(item)
    abstract = get_clean_abstract(item)
    return f"Title: {title}\nAbstract: {abstract}"


def get_clean_full_text(item: dict) -> str:
    return item["document"]


def get_clean_paragraph_indices(item: dict) -> List[Tuple[int, int]]:
    text = get_clean_full_text(item)
    paragraph_indices = []
    paragraph_start = 0
    paragraph_end = 0
    while paragraph_start < len(text):
        paragraph_end = text.find("\n\n", paragraph_start)
        if paragraph_end == -1:
            paragraph_end = len(text)
        paragraph_indices.append((paragraph_start, paragraph_end))
        paragraph_start = paragraph_end + 2
    return paragraph_indices


def get_clean_text(item: dict, start_idx: int, end_idx: int) -> str:
    text = get_clean_full_text(item)
    assert start_idx >= 0 and end_idx >= 0
    assert start_idx <= end_idx
    assert end_idx <= len(text)
    return text[start_idx:end_idx]


def get_clean_paragraphs(item: dict, min_words: int = 10) -> List[str]:
    text = get_clean_full_text(item)
    doc = nlp(text)
    paragraphs = [str(sent) for sent in doc.sents]
    paragraphs = [
        paragraph for paragraph in paragraphs if len(paragraph.split()) >= min_words
    ]
    return paragraphs


def get_clean_propositions(data: List[dict], batch_size: int = 16) -> List[str]:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    import torch
    import json

    model_name = "chentong00/propositionizer-wiki-flan-t5-large"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    def process_batch(paragraphs_batch):
        input_ids = tokenizer(
            paragraphs_batch, return_tensors="pt", padding=True, truncation=True
        ).input_ids
        outputs = model.generate(input_ids.to(device), max_new_tokens=512)
        return tokenizer.batch_decode(outputs, skip_special_tokens=True)

    propositions = []
    for record in tqdm(data, position=0, leave=True):
        paragraphs = get_clean_paragraphs(record)
        for i in tqdm(range(0, len(paragraphs), batch_size), position=1, leave=False):
            batch = paragraphs[i : i + batch_size]
            batch_outputs = process_batch(batch)
            propositions.extend(batch_outputs)
        break  # Remove this if you don't want to process only the first record
    return propositions


def get_clean_dict(data: Dataset) -> dict:
    return {get_clean_corpusid(item): item for item in data}


##### openai gpt-4 model #####


def get_gpt4_model(
    model_name: str = "gpt-4-1106-preview", azure: bool = True
) -> OPENAIBaseEngine:
    model = OPENAIBaseEngine(model_name, azure)
    model.test_api()
    return model


def prompt_gpt4_model(
    model: OPENAIBaseEngine, prompt: str = None, messages: List[dict] = None
) -> str:
    if prompt is not None:
        messages = [{"role": "assistant", "content": prompt}]
    elif messages is None:
        raise ValueError("Either prompt or messages must be provided")

    response = model.safe_completion(messages)
    if response["finish_reason"] != "stop":
        print(f"Unexpected stop reason: {response['finish_reason']}")
    return response["content"]


##### cache directory #####


def get_cache_dir() -> str:
    return os.environ["HF_HOME"]
