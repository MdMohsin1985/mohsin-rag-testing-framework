# AI RAG Testing Framework

A small Python project that ingests TXT, PDF, and Excel documents, stores chunks
in ChromaDB, answers questions with OpenAI, and evaluates responses with
DeepEval.

## Project Structure

```text
documents/
  sample.txt
datasets/
  evaluation_dataset.json
  hallucination_tests.json
src/
  ingest.py
  rag_app.py
  evaluate.py
  dataset_runner.py
  hallucination_test.py
rag_runs/
  latest_run.json
reports/
  evaluation_report.csv
  hallucination_report.csv
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

Add supported files to the `documents/` folder, then run:

```bash
python src/ingest.py
```

Supported file types:

- `.txt`
- `.pdf`
- `.xlsx`
- `.xlsm`
- `.xltx`
- `.xltm`

This creates a persistent ChromaDB database in `chroma_db/`.

Only files in `documents/` are ingested. Files in `datasets/` are test data and
should not be ingested into ChromaDB.

## Run the RAG App

Ask a question from the command line:

```bash
python src/rag_app.py --question "What does the AI RAG Testing Framework evaluate?"
```

The app retrieves relevant chunks from ChromaDB and asks OpenAI to answer using
only that context. It also saves the exact question, answer, and retrieved
context to `rag_runs/latest_run.json` plus a timestamped JSON file.

## Run Evaluation

Evaluate the latest saved RAG app answer:

```bash
python src/evaluate.py
```

Or evaluate a new question directly without using a saved JSON run:

```bash
python src/evaluate.py --question "What does the AI RAG Testing Framework evaluate?"
```

The evaluation uses:

- Faithfulness: Checks whether the generated answer is supported by the
  retrieved context and does not introduce unsupported claims.
- Answer Relevancy: Checks whether the generated answer directly addresses the
  user's question without adding off-topic information.
- Correctness: Compares the generated answer with the expected answer and checks
  whether they match in meaning, even if the wording is different.

These metrics use OpenAI through DeepEval, so `.env` must contain a valid
`OPENAI_API_KEY`.

## Run Dataset Evaluation

Run all question and expected-answer pairs from `datasets/evaluation_dataset.json`:

```bash
python src/dataset_runner.py
```

This creates `reports/evaluation_report.csv`.

## Run Hallucination Testing

Run unanswerable questions from `datasets/hallucination_tests.json`:

```bash
python src/hallucination_test.py
```

The hallucination dataset is not a source document. It should stay in
`datasets/` so the RAG system cannot retrieve the test questions as context.
The script flags hallucinated responses when DeepEval faithfulness fails and
saves results to `reports/hallucination_report.csv`.
