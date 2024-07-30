import json
import numpy as np
from sklearn.metrics import precision_score, recall_score

def load_data(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def calculate_context_precision(y_true, y_pred):
    return precision_score(y_true, y_pred, average='micro')

def calculate_context_recall(y_true, y_pred):
    return recall_score(y_true, y_pred, average='micro')

def calculate_context_relevance(y_true, y_pred):
    return np.mean([1 if pred in true else 0 for pred, true in zip(y_pred, y_true)])

def evaluate_retrieval(data):
    y_true = [item['relevant_contexts'] for item in data]
    y_pred = [item['retrieved_contexts'] for item in data]

    metrics = {
        'context_precision': calculate_context_precision(y_true, y_pred),
        'context_recall': calculate_context_recall(y_true, y_pred),
        'context_relevance': calculate_context_relevance(y_true, y_pred),
    }
    return metrics

if __name__ == "__main__":
    data_file = 'path/to/query-context-pairs.json'
    data = load_data(data_file)
    metrics = evaluate_retrieval(data)
    print(metrics)