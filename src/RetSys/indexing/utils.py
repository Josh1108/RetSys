import os
from typing import List, Any, Tuple
from datasets import Dataset
import spacy
from spacy.pipeline import Sentencizer
from tqdm import tqdm

nlp = spacy.blank("en")
nlp.add_pipe("sentencizer")
nlp.max_length = 100000000



##### reading fields from corpus_clean #####


def get_clean_corpusid(item: dict) -> str:
    if "page_number" in item:
        return f"{item['file_name']}_page_{item['page_number']}"
    else:
        return item["file_name"]


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
    return propositions


def get_clean_dict(data: Dataset) -> dict:
    return {get_clean_corpusid(item): item for item in data}


def get_cache_dir() -> str:
    return os.environ["HF_HOME"]
