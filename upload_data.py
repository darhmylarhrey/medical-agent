# import os
# import json
# import weaviate
# from dotenv import load_dotenv
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import Weaviate
# from langchain.docstore.document import Document
# from tqdm import tqdm

# load_dotenv()

# def create_schema(client: weaviate.Client):
#     schema = {
#         "classes": [
#             {
#                 "class": "PatientDocument",
#                 "description": "A document containing patient data",
#                 "properties": [
#                     {
#                         "dataType": ["string"],
#                         "description": "Unique identifier for the patient",
#                         "name": "patient_id"
#                     },
#                     {
#                         "dataType": ["string"],
#                         "description": "Unique user ID for the patient",
#                         "name": "patient_uid"
#                     },
#                     {
#                         "dataType": ["string"],
#                         "description": "PubMed ID for the related article",
#                         "name": "PMID"
#                     },
#                     {
#                         "dataType": ["string"],
#                         "description": "File path of the document",
#                         "name": "file_path"
#                     },
#                     {
#                         "dataType": ["string"],
#                         "description": "Title of the document",
#                         "name": "title"
#                     },
#                     {
#                         "dataType": ["text"],
#                         "description": "Content of the patient document",
#                         "name": "patient"
#                     },
#                     {
#                         "dataType": ["number"],
#                         "description": "Age of the patient",
#                         "name": "age"
#                     },
#                     {
#                         "dataType": ["string"],
#                         "description": "Gender of the patient",
#                         "name": "gender"
#                     },
#                     {
#                         "dataType": ["string[]"],
#                         "description": "Relevant articles associated with the patient",
#                         "name": "relevant_articles"
#                     },
#                     {
#                         "dataType": ["string[]"],
#                         "description": "Similar patients",
#                         "name": "similar_patients"
#                     }
#                 ]
#             }
#         ]
#     }
#     client.schema.create(schema)
#     print("Schema created successfully")

# def upload_data_to_weaviate(data_file: str, client: weaviate.Client, index_name: str, text_key: str, batch_size: int = 100):
#     with open(data_file, 'r') as file:
#         data = json.load(file)

#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     patient_texts = [patient[text_key] for patient in data]
#     text_chunks = text_splitter.split_text(' '.join(patient_texts))

#     documents = [Document(page_content=chunk) for chunk in text_chunks]

#     vector_store = Weaviate(client=client, index_name=index_name, text_key=text_key)

#     total_batches = len(documents) // batch_size + (1 if len(documents) % batch_size != 0 else 0)

#     for i in tqdm(range(0, len(documents), batch_size), desc="Uploading batches"):
#         batch = documents[i:i + batch_size]
#         vector_store.add_documents(batch)
#         print(f"Uploaded batch {i // batch_size + 1}/{total_batches}")

# if __name__ == "__main__":
#     weaviate_url = os.getenv("WEAVIATE_URL")
#     weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
#     data_file = "/Users/dhrey/Desktop/Workspace/NEU/INFO7375/testBot/data/PMC-Patients-mini.json"  # Replace with actual path

#     auth_config = weaviate.AuthApiKey(api_key=weaviate_api_key)
#     weaviate_client = weaviate.Client(url=weaviate_url, auth_client_secret=auth_config)

#     # Create the schema
#     create_schema(weaviate_client)

#     index_name = "PatientDocument"  # The index name defined in the schema
#     text_key = "patient"  # The key containing the main textual content

#     # Upload the data
#     upload_data_to_weaviate(data_file, weaviate_client, index_name, text_key)

# import os
# import json
# import weaviate
# from dotenv import load_dotenv
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import Weaviate
# from langchain.docstore.document import Document
# from tqdm import tqdm

# load_dotenv()

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# if not OPENAI_API_KEY:
#     raise ValueError("API Key not found in environment variables")

# def create_schema(client: weaviate.Client):
#     schema = {
#         "classes": [
#             {
#                 "class": "PubMedicalData",
#                 "description": "A document containing patient data",
#                 "vectorizer": "text2vec-openai",
#                  "moduleConfig": {
#                     "text2vec-openai": {
#                     "model": "text-embedding-3-large",
#                     "dimensions": 3072,
#                     "type": "text",
#                     "apiKey": OPENAI_API_KEY
#                     }
#                 },
#                 "properties": [
#                     {
#                         "dataType": ["string"],
#                         "description": "Unique identifier for the patient",
#                         "name": "patient_id"
#                     },
#                     {
#                         "dataType": ["string"],
#                         "description": "Unique user ID for the patient",
#                         "name": "patient_uid"
#                     },
#                     {
#                         "dataType": ["string"],
#                         "description": "PubMed ID for the related article",
#                         "name": "PMID"
#                     },
#                     {
#                         "dataType": ["string"],
#                         "description": "File path of the document",
#                         "name": "file_path"
#                     },
#                     {
#                         "dataType": ["string"],
#                         "description": "Title of the document",
#                         "name": "title"
#                     },
#                     {
#                         "dataType": ["text"],
#                         "description": "Content of the patient document",
#                         "name": "patient"
#                     },
#                     {
#                         "dataType": ["number"],
#                         "description": "Age of the patient",
#                         "name": "age"
#                     },
#                     {
#                         "dataType": ["string"],
#                         "description": "Gender of the patient",
#                         "name": "gender"
#                     },
#                     {
#                         "dataType": ["string[]"],
#                         "description": "Relevant articles associated with the patient",
#                         "name": "relevant_articles"
#                     },
#                     {
#                         "dataType": ["string[]"],
#                         "description": "Similar patients",
#                         "name": "similar_patients"
#                     }
#                 ]
#             }
#         ]
#     }
#     client.schema.create(schema)
#     print("Schema created successfully")

# def upload_data_to_weaviate(data_file: str, client: weaviate.Client, index_name: str, text_key: str, batch_size: int = 100):
#     # Load data from JSON file
#     with open(data_file, 'r') as file:
#         data = json.load(file)

#     # Initialize text splitter
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
#     # Collect all patient texts
#     patient_texts = []
#     for patient in data:
#         if text_key in patient and isinstance(patient[text_key], str):
#             patient_texts.append(patient[text_key])
#         else:
#             print(f"Skipping patient with id {patient.get('patient_id', 'unknown')} due to missing or invalid text key")

#     # Split the collected texts into chunks
#     text_chunks = text_splitter.split_text(' '.join(patient_texts))

#     # Create Document objects from the text chunks
#     documents = [Document(page_content=chunk) for chunk in text_chunks]

#     # Initialize Weaviate vector store
#     vector_store = Weaviate(client=client, index_name=index_name, text_key=text_key)

#     # Calculate total batches
#     total_batches = len(documents) // batch_size + (1 if len(documents) % batch_size != 0 else 0)

#     # Upload documents in batches
#     for i in tqdm(range(0, len(documents), batch_size), desc="Uploading batches"):
#         batch = documents[i:i + batch_size]
#         vector_store.add_documents(batch)
#         print(f"Uploaded batch {i // batch_size + 1}/{total_batches}")

# if __name__ == "__main__":
#     # Load environment variables
#     weaviate_url = os.getenv("WEAVIATE_URL")
#     weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
#     data_file = "/Users/dhrey/Desktop/Workspace/NEU/INFO7375/medical-agent/data/PMC-Patients-mini.json"  # Replace with actual path

#     # Initialize Weaviate client
#     auth_config = weaviate.AuthApiKey(api_key=weaviate_api_key)
#     weaviate_client = weaviate.Client(url=weaviate_url, auth_client_secret=auth_config)

#     # Set the API key in the environment variables for Weaviate
#     os.environ['OPENAI_APIKEY'] = OPENAI_API_KEY

#     # Create the schema in Weaviate
#     create_schema(weaviate_client)

#     index_name = "PubMedicalData"  # The index name defined in the schema
#     text_key = "patient"  # The key containing the main textual content

#     # Upload the data to Weaviate
#     upload_data_to_weaviate(data_file, weaviate_client, index_name, text_key)

# import os
# import json
# import weaviate
# from dotenv import load_dotenv
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import Weaviate
# from langchain.docstore.document import Document
# from tqdm import tqdm

# load_dotenv()

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# if not OPENAI_API_KEY:
#     raise ValueError("API Key not found in environment variables")

# def create_schema(client: weaviate.Client):
#     schema = {
#         "classes": [
#             {
#                 "class": "PubsMedicalData",
#                 "description": "A document containing patient data",
#                 "vectorizer": "text2vec-openai",
#                  "moduleConfig": {
#                     "text2vec-openai": {
#                     "model": "text-embedding-3-large",
#                     "dimensions": 3072,
#                     "type": "text",
#                     "apiKey": OPENAI_API_KEY
#                     }
#                 },
#                 "properties": [
#                     {
#                         "dataType": ["string"],
#                         "description": "Unique identifier for the patient",
#                         "name": "patient_id"
#                     },
#                     {
#                         "dataType": ["string"],
#                         "description": "Unique user ID for the patient",
#                         "name": "patient_uid"
#                     },
#                     {
#                         "dataType": ["string"],
#                         "description": "PubMed ID for the related article",
#                         "name": "PMID"
#                     },
#                     {
#                         "dataType": ["string"],
#                         "description": "File path of the document",
#                         "name": "file_path"
#                     },
#                     {
#                         "dataType": ["string"],
#                         "description": "Title of the document",
#                         "name": "title"
#                     },
#                     {
#                         "dataType": ["text"],
#                         "description": "Content of the patient document",
#                         "name": "patient"
#                     },
#                     {
#                         "dataType": ["number"],
#                         "description": "Age of the patient",
#                         "name": "age"
#                     },
#                     {
#                         "dataType": ["string"],
#                         "description": "Gender of the patient",
#                         "name": "gender"
#                     },
#                     {
#                         "dataType": ["string[]"],
#                         "description": "Relevant articles associated with the patient",
#                         "name": "relevant_articles"
#                     },
#                     {
#                         "dataType": ["string[]"],
#                         "description": "Similar patients",
#                         "name": "similar_patients"
#                     }
#                 ]
#             }
#         ]
#     }
#     client.schema.create(schema)
#     print("Schema created successfully")

# def upload_data_to_weaviate(data_file: str, client: weaviate.Client, index_name: str, text_key: str, batch_size: int = 100):
#     # Load data from JSON file
#     with open(data_file, 'r') as file:
#         data = json.load(file)

#     # Initialize text splitter
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
#     # Collect all patient texts
#     patient_texts = []
#     for patient in data:
#         if text_key in patient and isinstance(patient[text_key], str):
#             patient_texts.append(patient[text_key])
#         else:
#             print(f"Skipping patient with id {patient.get('patient_id', 'unknown')} due to missing or invalid text key")

#     # Split the collected texts into chunks
#     text_chunks = text_splitter.split_text(' '.join(patient_texts))

#     # Create Document objects from the text chunks
#     documents = [Document(page_content=chunk) for chunk in text_chunks]

#     # Initialize Weaviate vector store
#     vector_store = Weaviate(client=client, index_name=index_name, text_key=text_key)

#     # Calculate total batches
#     total_batches = len(documents) // batch_size + (1 if len(documents) % batch_size != 0 else 0)

#     # Upload documents in batches
#     for i in tqdm(range(0, len(documents), batch_size), desc="Uploading batches"):
#         batch = documents[i:i + batch_size]
#         vector_store.add_documents(batch)
#         print(f"Uploaded batch {i // batch_size + 1}/{total_batches}")

# if __name__ == "__main__":
#     # Load environment variables
#     weaviate_url = os.getenv("WEAVIATE_URL")
#     weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
#     data_file = "/Users/dhrey/Desktop/Workspace/NEU/INFO7375/medical-agent/data/PMC-Patients-mini.json"  # Replace with actual path

#     # Initialize Weaviate client
#     auth_config = weaviate.AuthApiKey(api_key=weaviate_api_key)
#     weaviate_client = weaviate.Client(url=weaviate_url, auth_client_secret=auth_config)

#     # Set the API key in the environment variables for Weaviate
#     os.environ['OPENAI_APIKEY'] = OPENAI_API_KEY

#     # Create the schema in Weaviate
#     create_schema(weaviate_client)

#     index_name = "PubsMedicalData"  # The index name defined in the schema
#     text_key = "patient"  # The key containing the main textual content

#     # Upload the data to Weaviate
#     upload_data_to_weaviate(data_file, weaviate_client, index_name, text_key)

# import os
# import json
# import weaviate
# import torch
# import time
# from dotenv import load_dotenv
# from transformers import AutoTokenizer, AutoModel
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from tqdm import tqdm

# load_dotenv()

# # Initialize the tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder")
# model = AutoModel.from_pretrained("ncbi/MedCPT-Article-Encoder")

# def encode_text(text):
#     inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
#     with torch.no_grad():
#         outputs = model(**inputs)
#     return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# def create_schema(client: weaviate.Client):
#     schema = {
#         "classes": [
#             {
#                 "class": "PubMedicData",
#                 "description": "A document containing patient data",
#                 "vectorizer": "none",  # No automatic vectorizer
#                 "properties": [
#                     {
#                         "dataType": ["string"],
#                         "description": "Unique identifier for the patient",
#                         "name": "patient_id"
#                     },
#                     {
#                         "dataType": ["string"],
#                         "description": "Unique user ID for the patient",
#                         "name": "patient_uid"
#                     },
#                     {
#                         "dataType": ["string"],
#                         "description": "PubMed ID for the related article",
#                         "name": "PMID"
#                     },
#                     {
#                         "dataType": ["string"],
#                         "description": "File path of the document",
#                         "name": "file_path"
#                     },
#                     {
#                         "dataType": ["string"],
#                         "description": "Title of the document",
#                         "name": "title"
#                     },
#                     {
#                         "dataType": ["text"],
#                         "description": "Content of the patient document",
#                         "name": "patient"
#                     },
#                     {
#                         "dataType": ["number"],
#                         "description": "Age of the patient",
#                         "name": "age"
#                     },
#                     {
#                         "dataType": ["string"],
#                         "description": "Gender of the patient",
#                         "name": "gender"
#                     },
#                     {
#                         "dataType": ["string[]"],
#                         "description": "Relevant articles associated with the patient",
#                         "name": "relevant_articles"
#                     },
#                     {
#                         "dataType": ["string[]"],
#                         "description": "Similar patients",
#                         "name": "similar_patients"
#                     }
#                 ]
#             }
#         ]
#     }
#     client.schema.create(schema)
#     print("Schema created successfully")

# def upload_data_to_weaviate(data_file: str, client: weaviate.Client, index_name: str, text_key: str, batch_size: int = 100):
#     # Load data from JSON file
#     with open(data_file, 'r') as file:
#         data = json.load(file)

#     # Initialize text splitter
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
#     # Collect all patient texts
#     patient_texts = []
#     for patient in data:
#         if text_key in patient and isinstance(patient[text_key], str):
#             patient_texts.append(patient[text_key])
#         else:
#             print(f"Skipping patient with id {patient.get('patient_id', 'unknown')} due to missing or invalid text key")

#     # Split the collected texts into chunks
#     text_chunks = text_splitter.split_text(' '.join(patient_texts))

#     # Create documents with embeddings
#     documents = []
#     for chunk in tqdm(text_chunks, desc="Encoding text chunks"):
#         embedding = encode_text(chunk).tolist()
#         document = {
#             "class": index_name,
#             "properties": {
#                 "patient": chunk,
#                 "vector": embedding
#             }
#         }
#         documents.append(document)

#     # Upload documents in batches with retry mechanism
#     total_batches = len(documents) // batch_size + (1 if len(documents) % batch_size != 0 else 0)
#     for i in tqdm(range(0, len(documents), batch_size), desc="Uploading batches"):
#         batch = documents[i:i + batch_size]
#         retry_count = 0
#         while retry_count < 5:
#             try:
#                 with client.batch(batch_size=batch_size) as batch_request:
#                     for doc in batch:
#                         batch_request.add_data_object(doc["properties"], doc["class"])
#                 print(f"Uploaded batch {i // batch_size + 1}/{total_batches}")
#                 break
#             except weaviate.exceptions.UnexpectedStatusCodeError as e:
#                 print(f"Error uploading batch {i // batch_size + 1}/{total_batches}: {e}")
#                 retry_count += 1
#                 wait_time = 2 ** retry_count
#                 print(f"Retrying in {wait_time} seconds...")
#                 time.sleep(wait_time)
#                 if retry_count == 5:
#                     print("Max retries reached. Skipping this batch.")

# if __name__ == "__main__":
#     # Load environment variables
#     weaviate_url = os.getenv("WEAVIATE_URL")
#     weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
#     data_file = "/Users/dhrey/Desktop/Workspace/NEU/INFO7375/medical-agent/data/PMC-Patients-mini.json"  # Replace with actual path

#     # Initialize Weaviate client
#     auth_config = weaviate.AuthApiKey(api_key=weaviate_api_key)
#     weaviate_client = weaviate.Client(url=weaviate_url, auth_client_secret=auth_config)

#     # Create the schema in Weaviate
#     create_schema(weaviate_client)

#     index_name = "PubMedicData"  # The index name defined in the schema
#     text_key = "patient"  # The key containing the main textual content

#     # Upload the data to Weaviate
#     upload_data_to_weaviate(data_file, weaviate_client, index_name, text_key)

import os
import json
import weaviate
import torch
import time
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm

load_dotenv()

# Initialize the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder")
model = AutoModel.from_pretrained("ncbi/MedCPT-Article-Encoder")

def encode_text(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def create_schema(client: weaviate.Client):
    schema = {
        "classes": [
            {
                "class": "PubMedicalData",
                "description": "A document containing patient data",
                # "vectorizer": "text2vec-openai",
                # "moduleConfig": {
                #     "text2vec-openai": {
                #         "model": "text-embedding-3-large",
                #         "dimensions": 3072,
                #         "type": "text"
                #     }
                # },
                "properties": [
                    {
                        "dataType": ["string"],
                        "description": "Unique identifier for the patient",
                        "name": "patient_id"
                    },
                    {
                        "dataType": ["string"],
                        "description": "Unique user ID for the patient",
                        "name": "patient_uid"
                    },
                    {
                        "dataType": ["string"],
                        "description": "PubMed ID for the related article",
                        "name": "PMID"
                    },
                    {
                        "dataType": ["string"],
                        "description": "File path of the document",
                        "name": "file_path"
                    },
                    {
                        "dataType": ["string"],
                        "description": "Title of the document",
                        "name": "title"
                    },
                    {
                        "dataType": ["text"],
                        "description": "Content of the patient document",
                        "name": "patient"
                    },
                    {
                        "dataType": ["number"],
                        "description": "Age of the patient",
                        "name": "age"
                    },
                    {
                        "dataType": ["string"],
                        "description": "Gender of the patient",
                        "name": "gender"
                    },
                    {
                        "dataType": ["string[]"],
                        "description": "Relevant articles associated with the patient",
                        "name": "relevant_articles"
                    },
                    {
                        "dataType": ["string[]"],
                        "description": "Similar patients",
                        "name": "similar_patients"
                    }
                ]
            }
        ]
    }
    client.schema.create(schema)
    print("Schema created successfully")

def upload_data_to_weaviate(data_file: str, client: weaviate.Client, index_name: str, text_key: str, batch_size: int = 500):
    # Load data from JSON file
    with open(data_file, 'r') as file:
        data = json.load(file)

    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    # Collect all patient texts
    patient_texts = []
    for patient in data:
        if text_key in patient and isinstance(patient[text_key], str):
            patient_texts.append(patient[text_key])
        else:
            print(f"Skipping patient with id {patient.get('patient_id', 'unknown')} due to missing or invalid text key")

    # Split the collected texts into chunks
    text_chunks = text_splitter.split_text(' '.join(patient_texts))

    # Create documents with embeddings
    documents = []
    for chunk in tqdm(text_chunks, desc="Encoding text chunks"):
        embedding = encode_text(chunk).tolist()
        document = {
            "class": index_name,
            "properties": {
                "patient": chunk,
                "vector": embedding
            }
        }
        documents.append(document)

    # Upload documents in batches with retry mechanism
    total_batches = len(documents) // batch_size + (1 if len(documents) % batch_size != 0 else 0)
    for i in tqdm(range(0, len(documents), batch_size), desc="Uploading batches"):
        batch = documents[i:i + batch_size]
        retry_count = 0
        while retry_count < 5:
            try:
                with client.batch(batch_size=batch_size) as batch_request:
                    for doc in batch:
                        batch_request.add_data_object(doc["properties"], doc["class"])
                print(f"Uploaded batch {i // batch_size + 1}/{total_batches}")
                break
            except weaviate.exceptions.UnexpectedStatusCodeError as e:
                print(f"Error uploading batch {i // batch_size + 1}/{total_batches}: {e}")
                retry_count += 1
                wait_time = 2 ** retry_count
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                if retry_count == 5:
                    print("Max retries reached. Skipping this batch.")

if __name__ == "__main__":
    # Load environment variables
    weaviate_url = os.getenv("WEAVIATE_URL")
    weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
    data_file = "/Users/dhrey/Desktop/Workspace/NEU/INFO7375/medical-agent/data/PMC-Patients-mini.json"  # Replace with actual path

    # Initialize Weaviate client
    auth_config = weaviate.AuthApiKey(api_key=weaviate_api_key)
    weaviate_client = weaviate.Client(url=weaviate_url, auth_client_secret=auth_config)

    # Create the schema in Weaviate
    create_schema(weaviate_client)

    index_name = "PubMedicalData"  # The index name defined in the schema
    text_key = "patient"  # The key containing the main textual content

    # Upload the data to Weaviate
    upload_data_to_weaviate(data_file, weaviate_client, index_name, text_key)