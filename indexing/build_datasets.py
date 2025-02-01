import os
import json
from datasets import Dataset
import PyPDF2

class DatasetConverter:
    def __init__(self):
        """
        Initialize the DatasetConverter with an empty list to store data.
        """
        self.data_list =[]

    def insert_json_file(self, json_file_path:str):
        """
        Insert a JSON file into the dataset.

        :param json_file_path: Path to the JSON file to be inserted.
        :type json_file_path: str
        """
        with open(json_file_path, "r") as f:
            json_data = json.load(f)  # Load JSON content
            combined_text = " ".join(json_data)  # Combine the list of texts
            self.data_list.append({"file_name": json_file_path, "document": combined_text})
    
    def parse_pdf_file(self, pdf_file_path:str):
        """
        Parse a PDF file and insert its content into the dataset.

        :param pdf_file_path: Path to the PDF file to be parsed.
        :type pdf_file_path: str
        """
        with open(pdf_file_path, "rb") as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page in pdf_reader.pages:
                text = page.extract_text()
                self.data_list.append({"file_name": pdf_file_path, "document": text})
    def load_dir_files(self, data_dir:str):
        """
        Load all files in a directory and insert them into the dataset.

        :param data_dir: Path to the directory containing files to be loaded.
        :type data_dir: str
        """
        for file in os.listdir(data_dir):
            if file.endswith(".json"):
                self.insert_json_file(os.path.join(data_dir, file))
            elif file.endswith(".pdf"):
                self.parse_pdf_file(os.path.join(data_dir, file))

    def save_dataset(self, dataset_name:str, private:bool=False,save_locally:bool=False,save_on_hf_hub:bool=False):
        """
        Save the dataset to a local file or push it to the Hugging Face Hub.

        :param dataset_name: Name of the dataset to be saved or pushed.
        :type dataset_name: str
        :param private: Whether the dataset should be private.
        :type private: bool
        :param save_locally: Whether to save the dataset locally.
        :type save_locally: bool
        :param save_on_hf_hub: Whether to push the dataset to the Hugging Face Hub.
        :type save_on_hf_hub: bool
        """
        assert save_locally or save_on_hf_hub, "Must save dataset locally or on HF Hub"
        processed_dataset = Dataset.from_list(self.data_list, split="full")
        if save_locally:
            processed_dataset.save_to_disk(dataset_name)
        elif save_on_hf_hub:
            processed_dataset.push_to_hub(dataset_name, private=private)
    
    def run(self, data_dir:str, dataset_name:str, private:bool=False,save_locally:bool=False,save_on_hf_hub:bool=False):
        """
        Run the dataset conversion and saving process.

        :param data_dir: Path to the directory containing files to be loaded.
        :type data_dir: str
        :param dataset_name: Name of the dataset to be saved or pushed.
        :type dataset_name: str
        :param private: Whether the dataset should be private.
        :type private: bool
        :param save_locally: Whether to save the dataset locally.
        :type save_locally: bool
        :param save_on_hf_hub: Whether to push the dataset to the Hugging Face Hub.
        :type save_on_hf_hub: bool
        """
        self.load_dir_files(data_dir)
        self.save_dataset(dataset_name, private, save_locally, save_on_hf_hub)
