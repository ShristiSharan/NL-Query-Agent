# NL-Query-Agent
Natural Language Query Agent to answer questions over a small set of lecture notes and a table of LLM architectures. Use of LLMs and open-source vector indexing and storage frameworks.


## Approach

1. **Data Collection**:
    - Gathered lecture notes from online sources.
    - Compiled a table of LLM architectures from research papers and articles.

2. **Data Organization**:
    - Structured lecture notes in JSON format.
    - Created a CSV file for LLM architectures.

3. **Indexing and Embedding**:
    - Used `BERT` from the `transformers` library to create embeddings.
    - Indexed the embeddings using `faiss` for efficient retrieval.

4. **Query Handling**:
    - Implemented a function to handle natural language queries and return relevant sections.

5. **Additional Features**:
    - Citing references for answers.
    - Basic conversational memory for follow-up questions.
    - Summary generation for conversation sessions.

## Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/ema-intern-challenge.git
    cd ema-intern-challenge
    ```

2. Set up the environment:
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

3. Run the application:
    ```bash
    python app.py
    ```

## Future Improvements

- Implement advanced conversational memory.
- Enhance citation functionality.
- Scale to handle multiple lectures and more complex queries.

## Dependencies

- transformers
- faiss-cpu
- pandas

