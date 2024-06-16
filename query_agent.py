import numpy as np
import faiss
import json
from transformers import AutoTokenizer, AutoModel
import torch
from preprocessing import cleaned_text

# Load pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

# Function to create embeddings for single text
def embed_text(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Load embeddings and texts
lecture_embeddings = np.load('embeddings.npy')
# -----loading sentences embedding
sentence_embeddings = np.load('sentence_embeddings.npy') 

with open('texts.json', 'r') as f:
    texts = json.load(f)

# Load lecture notes data
with open('lectures_notes.json', 'r') as f:
    lecture_data = json.load(f)
# Load sentences data


# Build FAISS index
index = faiss.IndexFlatIP(768)  # Assuming embedding size is 768
index.add(lecture_embeddings)

# Query function
def query(text):
    embedding = embed_text(text)
    _, indices = index.search(np.array([embedding]), k=1)
    result_index = indices[0][0]
    result = lecture_data['lectures'][indices[0][0]]
    return result

# Query function for sentences
def query_sentence(text):
    embedding = embed_text(text)
    _, indices = index.search(np.array([embedding]), k=1)
    sentence_index=indices[0][0]
    result=cleaned_text[sentence_index]
    return result

# Test queries
queries = [
    "What is neural network?",
    "what is machine learning?",
    "what are applications of Machine learning?"
]

# Perform queries and print results
for query_text in queries:
    print(f"Query: {query_text}")
    # result = query(query_text)   //either from quey function from json
    result=query_sentence(query_text)      
    print("Result:")
    print(result)
    print("------------------------")
