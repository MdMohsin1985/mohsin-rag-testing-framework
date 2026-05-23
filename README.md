# AI RAG Testing Framework

A small Python project that ingests TXT documents, stores chunks in ChromaDB,
answers questions with OpenAI, and evaluates responses with DeepEval.

## Project Structure

```text
documents/
  sample.txt
src/
  ingest.py
  rag_app.py
  evaluate.py
requirements.txt
.env.example
README.md
```

## Setup

1. Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create your environment file:

```bash
copy .env.example .env
```

4. Add your OpenAI API key to `.env`:

```text
OPENAI_API_KEY=your_real_openai_api_key_here
```

## Run Ingestion

Add `.txt` files to the `documents/` folder, then run:

```bash
python src/ingest.py
```

This creates a persistent ChromaDB database in `chroma_db/`.

## Run the RAG App

Ask a question from the command line:

```bash
python src/rag_app.py --question "What does the AI RAG Testing Framework evaluate?"
```

The app retrieves relevant chunks from ChromaDB and asks OpenAI to answer using
only that context.

## Run Evaluation

Run DeepEval metrics:

```bash
python src/evaluate.py
```

The evaluation uses:

- Faithfulness
- Answer Relevancy

Both metrics use OpenAI through DeepEval, so `.env` must contain a valid
`OPENAI_API_KEY`.
