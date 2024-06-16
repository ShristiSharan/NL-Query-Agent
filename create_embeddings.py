import json
import pandas as pd
import faiss
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

# Load pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

# Function to create embeddings
def embed_text(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Load lecture notes data
with open('lecture_notes.json', 'r') as f:
    lecture_data = json.load(f)

# Create FAISS index
d = 768  # Dimension of embeddings
index = faiss.IndexFlatL2(d)

# Embed and add lecture notes to index
lecture_embeddings = []
for lecture in lecture_data['lectures']:
    embedding = embed_text(lecture['content'])
    lecture_embeddings.append(embedding)
    index.add(np.array([embedding]))

# Save the index and embeddings
faiss.write_index(index, 'lecture_notes.index')
np.save('lecture_embeddings.npy', np.array(lecture_embeddings))

# Load LLM architectures data
llm_df = pd.read_csv('llm_architectures.csv')

# Create embeddings for LLM descriptions
llm_embeddings = []
for _, row in llm_df.iterrows():
    description = f"{row['Model Name']} {row['Year']} {row['Architecture Type']} {row['Key Features']}"
    embedding = embed_text(description)
    llm_embeddings.append(embedding)
    index.add(np.array([embedding]))

# Save the LLM embeddings
np.save('llm_embeddings.npy', np.array(llm_embeddings))
