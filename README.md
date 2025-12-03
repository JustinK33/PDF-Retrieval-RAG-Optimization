# Advanced PDF Retrieval & RAG Optimization with LlamaIndex and Gemini

A notebook-based project that turns a raw mortgage PDF into a searchable, question-answering system using Retrieval-Augmented Generation (RAG). It ingests the PDF, indexes it with LlamaIndex, experiments with multiple retrieval strategies, and uses Google Gemini to generate grounded answers.

---

## Tech Stack

- Python
- Jupyter Notebook (VS Code)
- LlamaIndex
- Google Gemini API
- HuggingFace Sentence Transformers
- PyMuPDF (`fitz`)
- Pandas + Matplotlib

---

## Features

- **End-to-end RAG pipeline**
  - Load a mortgage “Lender Fees Worksheet” PDF
  - Convert PDF pages to LlamaIndex `Document` objects with metadata
  - Build a `VectorStoreIndex` over the document text

- **PDF ingestion and parsing**
  - File selection from the local filesystem
  - Text extraction with PyMuPDF
  - Per-page metadata: file name, page number, total pages

- **Multiple retrieval strategies**
  - Semantic **vector retrieval** (embeddings)
  - **BM25 keyword retrieval** over the same nodes
  - **Hybrid retriever** that merges vector + BM25 results and removes duplicates

- **Query expansion and fusion**
  - Gemini-powered query expansion to rephrase the user’s question
  - Query fusion retriever that mixes results from multiple reformulated queries

- **Reranking with cross-encoders**
  - SentenceTransformer cross-encoder (`ms-marco-MiniLM-L-6-v2`)
  - Reranks candidate chunks by how well they match the query
  - Keeps only the top-N most relevant nodes for the final answer

- **Evaluation and visualization**
  - Side-by-side comparison of vector, BM25, and hybrid retrieval
  - DataFrames of results with scores and page numbers
  - Bar charts of retrieval scores to see how methods differ

- **Real Q&A over the mortgage PDF**
  - Answers domain-specific questions such as:
    - “What is the total estimated monthly payment?”
    - “How much does the borrower pay for lender’s title insurance?”

---

## What I Learned From This Project

- **Designing a RAG pipeline instead of just calling an LLM**  
  I learned how to separate the pipeline into ingestion, indexing, retrieval, reranking, and generation, and how each layer affects answer quality.

- **Working with LlamaIndex abstractions**  
  I explored how `Document`s, `VectorStoreIndex`, retrievers, and post-processors fit together, and how to swap components (embeddings, retrievers, rerankers) without rewriting everything.

- **Semantic vs keyword vs hybrid retrieval**  
  By comparing vector search, BM25, and a hybrid approach, I saw concrete trade-offs between semantic similarity and exact keyword matching on legal/financial text.

- **Using rerankers to improve answer grounding**  
  I learned how a cross-encoder reranker can significantly improve relevance by re-scoring small sets of candidate chunks before they reach the LLM.

- **Managing secrets and environments for AI projects**  
  I set up a `.env` file, used `python-dotenv` to load `GOOGLE_API_KEY`, and added `.env` and sample PDFs to `.gitignore` so the project is safe to share on GitHub.

---

## Set up Jupyter Notebook
1. mkdir RAG
2. cd RAG
3. python -m venv venv
4. source venv/bin/activate
5. create a file with .ipynb
6. select kernal (choose the venv you created in steps 3 & 4)
