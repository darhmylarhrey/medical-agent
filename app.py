# import os
# import time
# import json
# import weaviate
# import streamlit as st
# from dotenv import load_dotenv
# from requests.exceptions import HTTPError
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import Weaviate
# from langchain.prompts import PromptTemplate
# from langchain.chat_models import ChatOpenAI
# from langchain.chains.question_answering import load_qa_chain
# from langchain.docstore.document import Document

# load_dotenv()

# class RAGSystem:
#     def __init__(self, openai_api_key: str, weaviate_client: weaviate.Client, data_file: str):
#         self.openai_api_key = openai_api_key
#         self.weaviate_client = weaviate_client
#         self.data_file = data_file
#         self.vector_store = None
#         self.chat_history = [{'role': 'system', 'content': "Assistant is a large language model trained by OpenAI."}]
#         self.data = self.load_data()

#     def load_data(self):
#         try:
#             with open(self.data_file, 'r') as file:
#                 data = json.load(file)
#             return data
#         except Exception as e:
#             st.error(f"Error loading data: {e}")
#             return None

#     def get_text_chunks(self, text: str):
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#         return text_splitter.split_text(text)

#     def create_documents(self, texts: list):
#         return [Document(page_content=text) for text in texts]

#     def get_vector_store(self, text_chunks: list):
#         documents = self.create_documents(text_chunks)
#         try:
#             vector_store = Weaviate.from_documents(documents, client=self.weaviate_client)
#             return vector_store
#         except HTTPError as e:
#             st.error(f"HTTP error initializing vector store: {e}")
#             return None
#         except Exception as e:
#             st.error(f"Error initializing vector store: {e}")
#             return None

#     def get_conversational_chain(self):
#         try:
#             prompt_template = """
#             You are a knowledgeable medical assistant. Provide helpful and relevant advice and diagnosis based on the given context.
#             Context:\n {context}?\n
#             Symptoms: \n{symptoms}\n
#             Advice and Diagnosis:
#             """
#             model = ChatOpenAI(model="gpt-4", openai_api_key=self.openai_api_key)
#             prompt = PromptTemplate(template=prompt_template, input_variables=["context", "symptoms"])
#             return load_qa_chain(model, chain_type="stuff", prompt=prompt)
#         except Exception as e:
#             st.error(f"Error creating conversational chain: {e}")

#     def user_input(self, symptoms: str):
#         if self.vector_store is None:
#             raise Exception("RAG system has not been initialized. Please run initialize_rag first.")
#         self.chat_history.append({'role': 'user', 'content': symptoms})
#         try:
#             docs = self.vector_store.similarity_search(symptoms, k=5)
#             if not docs:
#                 response = {"output_text": "I have no knowledge about this based on the data I was trained on."}
#             else:
#                 chain = self.get_conversational_chain()
#                 response = chain({"input_documents": docs, "symptoms": symptoms}, return_only_outputs=True)
#             self.chat_history.append(response['output_text'])
#             return response
#         except Exception as e:
#             st.error(f"Error processing user input: {e}")
#             return {"output_text": "I have no knowledge about this based on the data I was trained on."}

#     def initialize_rag(self):
#         try:
#             patient_texts = [patient['patient'] for patient in self.data]
#             text_chunks = self.get_text_chunks(' '.join(patient_texts))
#             self.vector_store = self.get_vector_store(text_chunks)
#         except Exception as e:
#             st.error(f"Error initializing RAG system: {e}")

# # Streamlit interface
# def main():
#     st.title("Medical Assistant RAG System")
#     st.write("Developed by Abobarin Afeez")

#     openai_api_key = os.getenv("OPENAI_API_KEY")
#     weaviate_url = os.getenv("WEAVIATE_URL")
#     weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
#     data_file = "/Users/dhrey/Desktop/Workspace/NEU/INFO7375/testBot/data/PMC-Patients-mini.json"  # Replace with actual path

#     auth_config = weaviate.AuthApiKey(api_key=weaviate_api_key)
#     weaviate_client = weaviate.Client(url=weaviate_url, auth_client_secret=auth_config)

#     rag_system = RAGSystem(openai_api_key=openai_api_key, weaviate_client=weaviate_client, data_file=data_file)
#     rag_system.initialize_rag()
#     st.success("RAG system initialized successfully!")

#     symptoms = st.text_input("Enter your symptoms")

#     if st.button("Get Advice and Diagnosis"):
#         response = rag_system.user_input(symptoms)
#         st.write(response['output_text'])

# if __name__ == "__main__":
#     main()

# import os
# import streamlit as st
# from dotenv import load_dotenv
# from langchain.vectorstores import Weaviate as WeaviateVectorStore
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate
# from langchain.chat_models import ChatOpenAI
# from langchain.docstore.document import Document
# import weaviate

# load_dotenv()

# class RAGSystem:
#     def __init__(self):
#         self.vector_store = None
#         self.initialize_weaviate_client()
#         self.chat_history = [{'role': 'system', 'content': "Assistant is a large language model trained by OpenAI."}]

#     def initialize_weaviate_client(self):
#         weaviate_url = os.getenv("WEAVIATE_URL")
#         api_key = os.getenv("WEAVIATE_API_KEY")
        
#         auth_config = weaviate.AuthApiKey(api_key=api_key)
#         self.client = weaviate.Client(url=weaviate_url, auth_client_secret=auth_config)
#         self.vector_store = WeaviateVectorStore(client=self.client, index_name="PatientDocument", text_key="patient")
#         st.success("Weaviate client initialized successfully!")

#     def get_conversational_chain(self):
#         try:
#             prompt_template = """
#             You are a knowledgeable medical assistant. Provide helpful and relevant advice and diagnosis based on the given context.
#             Context:\n {context}?\n
#             Symptoms: \n{symptoms}\n
#             Advice and Diagnosis:
#             """
#             model = ChatOpenAI(model="gpt-4", openai_api_key=os.getenv("OPENAI_API_KEY"))
#             prompt = PromptTemplate(template=prompt_template, input_variables=["context", "symptoms"])
#             return load_qa_chain(model, chain_type="stuff", prompt=prompt)
#         except Exception as e:
#             st.error(f"Error creating conversational chain: {e}")

#     def user_input(self, symptoms: str):
#         if self.vector_store is None:
#             raise Exception("RAG system has not been initialized. Please run initialize_rag first.")
        
#         self.chat_history.append({'role': 'user', 'content': symptoms})
#         try:
#             # Perform hybrid search with both nearText and hybrid
#             search_result = self.client.query.get("PatientDocument", ["patient", "title", "file_path"]) \
#                 .with_near_text({"concepts": [symptoms]}) \
#                 .with_hybrid({"query": symptoms}) \
#                 .with_limit(5) \
#                 .do()
            
#             docs = search_result['data']['Get']['PatientDocument']
            
#             if not docs:
#                 response = {"output_text": "I have no knowledge about this based on the data I was trained on."}
#             else:
#                 chain = self.get_conversational_chain()
#                 input_docs = [Document(page_content=doc['patient']) for doc in docs if 'patient' in doc]
#                 response = chain({"input_documents": input_docs, "symptoms": symptoms}, return_only_outputs=True)
            
#             self.chat_history.append({'role': 'assistant', 'content': response['output_text']})
#             return response
#         except Exception as e:
#             st.error(f"Error processing user input: {e}")
#             return {"output_text": "I have no knowledge about this based on the data I was trained on."}

# # Streamlit interface
# def main():
#     st.title("Medical Assistant RAG System")
#     st.write("Developed by Abobarin Afeez")

#     rag_system = RAGSystem()

#     symptoms = st.text_input("Enter your symptoms")

#     if st.button("Get Advice and Diagnosis"):
#         response = rag_system.user_input(symptoms)
#         st.write(response['output_text'])

# if __name__ == "__main__":
#     main()


import os
import streamlit as st
from dotenv import load_dotenv
from langchain.vectorstores import Weaviate as WeaviateVectorStore
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
import weaviate

load_dotenv()

class RAGSystem:
    def __init__(self):
        self.vector_store = None
        self.initialize_weaviate_client()
        self.chat_history = [{'role': 'system', 'content': "Assistant is a large language model trained by OpenAI."}]

    def initialize_weaviate_client(self):
        weaviate_url = os.getenv("WEAVIATE_URL")
        api_key = os.getenv("WEAVIATE_API_KEY")
        
        auth_config = weaviate.AuthApiKey(api_key=api_key)
        self.client = weaviate.Client(url=weaviate_url, auth_client_secret=auth_config)
        self.vector_store = WeaviateVectorStore(client=self.client, index_name="PatientDocument", text_key="patient")
        st.success("Weaviate client initialized successfully!")

    def get_conversational_chain(self):
        try:
            prompt_template = """
            You are a knowledgeable medical assistant. Provide helpful and relevant advice and diagnosis based on the given context.
            Context:\n {context}?\n
            Symptoms: \n{symptoms}\n
            Advice and Diagnosis:
            """
            model = ChatOpenAI(model="gpt-4", openai_api_key=os.getenv("OPENAI_API_KEY"))
            prompt = PromptTemplate(template=prompt_template, input_variables=["context", "symptoms"])
            return load_qa_chain(model, chain_type="stuff", prompt=prompt)
        except Exception as e:
            st.error(f"Error creating conversational chain: {e}")

    def user_input(self, symptoms: str):
        if self.vector_store is None:
            raise Exception("RAG system has not been initialized. Please run initialize_rag first.")
        
        self.chat_history.append({'role': 'user', 'content': symptoms})
        try:
            # Perform hybrid search with both nearText and hybrid
            st.write("Attempting to perform hybrid search...")

            try:
                st.write("Attempting to perform hybrid search...")
                search_result = self.client.query.get("PatientDocument", ["patient", "title", "file_path"]) \
                    .with_hybrid(query=symptoms, alpha=0.5) \
                    .with_limit(5) \
                    .do()
                st.write("Search Result:", search_result)
            except Exception as search_exception:
                st.write(f"Error during search: {search_exception}")
                st.write("Search Query Debug Info:")
                st.write("Symptoms:", symptoms)
                st.write("Search Query:", {
                    "query": symptoms,
                    "alpha": 0.5,
                    "limit": 5
                })
                raise search_exception

            st.write("Search Result Raw:", search_result)

            # Ensure we are accessing the correct part of the response
            if 'data' in search_result and 'Get' in search_result['data'] and 'PatientDocument' in search_result['data']['Get']:
                docs = search_result['data']['Get']['PatientDocument']
            else:
                docs = []
            
            st.write("Extracted Docs:", docs)

            if not docs:
                response = {"output_text": "I have no knowledge about this based on the data I was trained on."}
            else:
                chain = self.get_conversational_chain()
                st.write("Input Docs for Chain:", [doc.get('patient', '') for doc in docs])
                input_docs = [Document(page_content=doc.get('patient', '')) for doc in docs]
                response = chain({"input_documents": input_docs, "symptoms": symptoms}, return_only_outputs=True)
            
            st.write("Response:", response)
            
            self.chat_history.append({'role': 'assistant', 'content': response['output_text']})
            return response
        except Exception as e:
            st.error(f"Error processing user input: {e}")
            return {"output_text": "I have no knowledge about this based on the data I was trained on."}

# Streamlit interface
def main():
    st.title("Medical Assistant RAG System")
    st.write("Developed by Abobarin Afeez")

    rag_system = RAGSystem()

    symptoms = st.text_input("Enter your symptoms")

    if st.button("Get Advice and Diagnosis"):
        response = rag_system.user_input(symptoms)
        st.write(response['output_text'])

if __name__ == "__main__":
    main()


    # a 60 years old man with fever, dry cough, and dyspnea