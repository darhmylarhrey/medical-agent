
### README: Setting Up and Using the Metrics Test Code

## Introduction

This README provides step-by-step instructions on setting up and using the test code to evaluate the performance metrics of the RAG-based medical assistant chatbot.

## Prerequisites

Before running the evaluation code, ensure you have the following prerequisites installed:

- Python 3.8+
- Install Required Python packages 
- Access to Weaviate instance
- Pre-trained MedCPT model from Hugging Face
- Dataset of queries and contexts

## Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/darhmylarhrey/medical-agent.git
cd medical-agent
```

### Step 2: Install Dependencies

Install the required Python packages:

### Step 3: Set Up Environment Variables

Create a `.env` file in the root directory and add the following environment variables:

```bash
WEAVIATE_URL=https://your-weaviate-instance-url
WEAVIATE_API_KEY=your-weaviate-api-key
OPENAI_API_KEY=your-openai-api-key
```


### Step 4: Encode and Upload Data to Weaviate

Prepare your data file (e.g., `PMC-Patients-mini.json`) and upload it to Weaviate:

```bash
python upload_data.py --data_file path/to/PMC-Patients-mini.json
```

## Running the Evaluation

### Step 1: Calculate Retrieval Metrics

Run the retrieval metrics evaluation script:

```bash
python evaluate_retrieval.py --data_file path/to/query-context-pairs.json
```

This script will calculate and print the following metrics:

- Context Precision
- Context Recall
- Context Relevance

### Step 2: Calculate Generation Metrics

Run the generation metrics evaluation script:

```bash
python evaluate_generation.py --data_file path/to/query-answer-pairs.json
```

This script will calculate and print the following metrics:

- Faithfulness
- Answer Relevance
- Information Integration


## Report

### Generating the Report

After calculating the metrics and implementing improvements, generate a detailed report:

```bash
python evaluate_metrics.py
```

This will create a comprehensive report comparing the performance before and after the improvements.


## Conclusion

This README provides a step-by-step guide to setting up and using the test code to evaluate the performance metrics of the RAG-based medical assistant chatbot. Follow the instructions carefully to reproduce the results and make further improvements to the system. For any issues or questions, please refer to the documentation or contact the project maintainers.