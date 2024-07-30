import os
import json
import weaviate
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

load_dotenv()

# Initialize the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder")
model = AutoModel.from_pretrained("ncbi/MedCPT-Article-Encoder")

def encode_text(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def calculate_retrieval_metrics(retrieved_contexts, relevant_contexts):
    precision = precision_score(relevant_contexts, retrieved_contexts, average='binary')
    recall = recall_score(relevant_contexts, retrieved_contexts, average='binary')
    f1 = f1_score(relevant_contexts, retrieved_contexts, average='binary')
    return precision, recall, f1

def calculate_generation_metrics(generated_answers, ground_truths):
    faithfulness = np.mean([1 if answer == ground_truth else 0 for answer, ground_truth in zip(generated_answers, ground_truths)])
    relevance_scores = [relevance_score(answer, query) for answer, query in zip(generated_answers, queries)]
    average_relevance = np.mean(relevance_scores)
    return faithfulness, average_relevance

def relevance_score(answer, query):
    pass

retrieved_contexts = [1, 0, 1, 1, 0]
relevant_contexts = [1, 1, 1, 0, 0]
precision, recall, f1 = calculate_retrieval_metrics(retrieved_contexts, relevant_contexts)
print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

generated_answers = ["answer1", "answer2", "answer3"]
ground_truths = ["answer1", "wrong_answer", "answer3"]
faithfulness, average_relevance = calculate_generation_metrics(generated_answers, ground_truths)
print(f"Faithfulness: {faithfulness}, Average Relevance: {average_relevance}")