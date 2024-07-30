import json
from sklearn.metrics import precision_score, recall_score

# Load query-context pairs
with open('data/query-context-pairs.json', 'r') as f:
    query_context_pairs = json.load(f)

# Load query-answer pairs
with open('data/query-answer-pairs.json', 'r') as f:
    query_answer_pairs = json.load(f)

# Function to calculate retrieval metrics
def calculate_retrieval_metrics(query_context_pairs):
    context_precision = []
    context_recall = []
    context_relevance = []
    
    for pair in query_context_pairs:
        retrieved = set(pair['retrieved_contexts'])
        relevant = set(pair['relevant_contexts'])
        
        true_positive = len(retrieved & relevant)
        false_positive = len(retrieved - relevant)
        false_negative = len(relevant - retrieved)
        
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        
        context_precision.append(precision)
        context_recall.append(recall)
        context_relevance.append(true_positive / len(relevant) if len(relevant) > 0 else 0)
    
    return {
        "context_precision": sum(context_precision) / len(context_precision),
        "context_recall": sum(context_recall) / len(context_recall),
        "context_relevance": sum(context_relevance) / len(context_relevance)
    }

# Function to calculate generation metrics
def calculate_generation_metrics(query_answer_pairs):
    faithfulness = []
    answer_relevance = []
    information_integration = []
    
    for pair in query_answer_pairs:
        expected = pair['expected_answer']
        generated = pair['generated_answer']
        
        faithfulness.append(int(expected == generated))
        answer_relevance.append(int(expected in generated))
        information_integration.append(int(len(expected.split()) == len(generated.split())))
    
    return {
        "faithfulness": sum(faithfulness) / len(faithfulness),
        "answer_relevance": sum(answer_relevance) / len(answer_relevance),
        "information_integration": sum(information_integration) / len(information_integration)
    }

# Calculate metrics
retrieval_metrics = calculate_retrieval_metrics(query_context_pairs)
generation_metrics = calculate_generation_metrics(query_answer_pairs)

# Combine metrics
metrics = {**retrieval_metrics, **generation_metrics}

# Save metrics
with open('metrics/before_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)

print("Metrics calculated and saved successfully.")