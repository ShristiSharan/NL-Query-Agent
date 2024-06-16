import numpy as np
import faiss
import json
from transformers import AutoTokenizer, AutoModel
import torch

# Load pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

# Function to create embeddings
def embed_text(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Load FAISS index and embeddings
index = faiss.read_index('lecture_notes.index')
lecture_embeddings = np.load('lecture_embeddings.npy')
llm_embeddings = np.load('llm_embeddings.npy')

# Load lecture notes data
with open('lecture_notes.json', 'r') as f:
    lecture_data = json.load(f)

# Query function
def query(text):
    embedding = embed_text(text)
    _, indices = index.search(np.array([embedding]), k=1)
    result = lecture_data['lectures'][indices[0][0]]
    return result

# Example query
result = query("What are the layers in a transformer block?")
print(result)
