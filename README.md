# NL-Query-Agent
Natural Language Query Agent to answer questions over a small set of lecture notes and a table of LLM architectures. Use of LLMs and open-source vector indexing and storage frameworks.

## Features

- Query lecture notes using natural language queries
- Retrieve specific answers to questions from lecture notes
- Process raw text data into structured formats
- Create embeddings for both lecture notes and raw text data
- Store embeddings and text data for efficient querying

## Project Structure

- `preprocessing.py`: Contains functions for preprocessing raw text data and organizing it into structured formats.
- `create_embeddings.py`: Creates embeddings for both lecture notes and raw text data.
- `query_agent.py`: Allows querying of lecture notes and raw text data using natural language queries.
- `lecture_notes.json`: Structured JSON file containing lecture notes data.
- `texts.json`: Structured JSON file containing raw text data.
- `sentence_embeddings.npy`: NumPy file containing embeddings for sentences.
- `sentence_index.index`: FAISS index file for fast similarity search.
- `README.md`: Instructions and overview of the project.
  
## Approach

1. **Data Collection**:
    - Gathered lecture notes from online sources.
    - Compiled a table of LLM architectures from research papers and articles.

2. **Data Organization**:
    - Structured lecture notes in JSON format and others preprocessed the Raw data.
    - Created a CSV file for LLM architectures.

3. **Indexing and Embedding**:
    - Used `BERT` from the `transformers` library to create embeddings.
    - Indexed the embeddings using `faiss` for efficient retrieval.

4. **Query Handling**:
    - Implemented a function to handle natural language queries and return relevant sections.

## Setup

1. Fork and Clone the repository:
    ```bash
    git clone https://github.com/your-username/NL-Query-Agent.git
    cd NL-Query-Agent
    ```

2. Set up the environment:
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

3. Run the application:
    ```bash
    python query_agent.py

    ```

## Future Improvements

- Implement advanced conversational memory.
- Enhance citation functionality.
- Scale to handle multiple lectures and more complex queries.

## Requirements

- Python 3.x
- PyTorch
- Transformers library
- Faiss
- NLTK


This README provides an overview of the project, its structure, processes, and how to use it. Make sure to replace `yourusername` with your actual GitHub username.
