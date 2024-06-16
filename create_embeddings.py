import json
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch


# Define the actual model name you will use for generating embeddings
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

# # Function to generate embeddings for a given text
# def generate_embeddings(text, tokenizer, model):
#     inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
#     with torch.no_grad():
#         outputs = model(**inputs)
#     return outputs.last_hidden_state.mean(dim=1).numpy()

# # Load lecture notes
# with open('lectures_notes.json', 'r') as f:
#     lecture_data = json.load(f)

# lecture_notes = lecture_data['lectures']

# # Load LLM architectures
# llm_architectures = pd.read_csv('llm_architectures.csv')

# # Combine the data into a single list of texts to embed
# texts = []

# # Extract text from lecture notes
# for lecture in lecture_notes:
#     texts.append(lecture['content'])  
# # Extract text from LLM architectures
# for _, row in llm_architectures.iterrows():
#     texts.append(row['Key Features'])  

# # Generate embeddings for all texts
# embeddings = []
# for text in texts:
#     embeddings.append(generate_embeddings(text, tokenizer, model))

# embeddings = np.vstack(embeddings)

# # Save the embeddings and associated texts
# np.save('embeddings.npy', embeddings)
# with open('texts.json', 'w') as f:
#     json.dump(texts, f)

# print("Embeddings and texts have been saved.")

# ------------------------The Above approach is for json file structure notes
# -----------And below is creating embedding of raw data which got preprocesed

from transformers import AutoTokenizer, AutoModel
from preprocessing import sentences

# Function to create embeddings for a list of sentences
def create_embeddings(sentences):
    embeddings = []
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        # Take the mean of the last hidden state across tokens to get sentence embedding
        sentence_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        embeddings.append(sentence_embedding)
    return np.array(embeddings)

# Create embeddings for tokenized sentences
sentence_embeddings = create_embeddings(sentences)

# Save embeddings
np.save('sentence_embeddings.npy', sentence_embeddings)