import os
import json
import argparse
from datasets import Dataset


def push_dataset_to_hub(data_dir, dataset_name, private=False):
    # Directory containing JSON files
    data_dir = "/home/jkalra/directed_study/current_databases/docs/jsons"

    # List all JSON files in the directory
    json_files = [f for f in os.listdir(data_dir) if f.endswith(".json")]

    data = []
    for file_name in json_files:
        file_path = os.path.join(data_dir, file_name)
        with open(file_path, "r") as f:
            json_data = json.load(f)  # Load JSON content
            combined_text = " ".join(json_data)  # Combine the list of texts
            data.append({"file_name": file_name, "document": combined_text})

    processed_dataset = Dataset.from_list(data, split="full")

    processed_dataset.push_to_hub(dataset_name, private=True)


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", required=True)
parser.add_argument("--index_name", required=True)
args = parser.parse_args()
push_dataset_to_hub(args.data_dir, args.index_name)
