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

    def __parse_json_file(self, json_data:str):
        """Parses a JSON object containing key-value lists into a formatted string."""
        if isinstance(json_data, str):  # If input is a string, parse it into a dictionary
            json_data = json.loads(json_data)
    
        def recursive_parse(data, prefix=""):
            result = []
            if isinstance(data, dict):
                for key, value in data.items():
                    result.append(recursive_parse(value, prefix + key + ": "))
            elif isinstance(data, list):
                for item in data:
                    result.append(recursive_parse(item, prefix + "- "))
            else:
                result.append(f"{prefix}{str(data)}")
            return "\n".join(filter(None, result))
    
        return recursive_parse(json_data)

    def insert_json_file(self, json_file_path:str):
        """
        Insert a JSON file into the dataset.

        :param json_file_path: Path to the JSON file to be inserted.
        :type json_file_path: str
        """
        with open(json_file_path, "r") as f:
            json_data = json.load(f)  # Load JSON content
            combined_text = self.__parse_json_file(json_data) 
            self.data_list.append({"file_name": json_file_path, "document": combined_text})
    
    def parse_pdf_file(self, pdf_file_path:str):
        """
        Parse a PDF file and insert its content into the dataset.

        :param pdf_file_path: Path to the PDF file to be parsed.
        :type pdf_file_path: str
        """
        with open(pdf_file_path, "rb") as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for i,page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                self.data_list.append({"file_name": pdf_file_path, "document": text, "page_number": i+1})
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

    def save_dataset(self, dataset_name:str, private:bool=False,save_locally:bool=False,save_on_hf_hub:bool=False,dataset_dir:str="."):
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
        :param dataset_dir: Path to the dataset to be saved or pushed.
        :type dataset_dir: str
        """
        assert save_locally or save_on_hf_hub, "Must save dataset locally or on HF Hub"
        if save_locally:
            dataset_path = os.path.join(dataset_dir, dataset_name)
        processed_dataset = Dataset.from_list(self.data_list, split="full")
        if save_locally:
            processed_dataset.save_to_disk(dataset_path)
        if save_on_hf_hub:
            processed_dataset.push_to_hub(dataset_name, private=private)
    
    def run(self, data_dir:str, dataset_name:str, private:bool=False,save_locally:bool=False,save_on_hf_hub:bool=False,dataset_dir:str="."):
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
        :param dataset_dir: Path to the dataset to be saved or pushed.
        :type dataset_dir: str
        """
        self.load_dir_files(data_dir)
        self.save_dataset(dataset_name, private, save_locally, save_on_hf_hub,dataset_dir)
