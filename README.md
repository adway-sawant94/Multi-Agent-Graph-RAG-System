# Mini-Graph-RAG: Multi-Agent Relation-Finding Knowledge Graph Agent

Mini-Graph-RAG is a high-performance, interview-grade implementation of a **Multi-Agent Graph Retrieval-Augmented Generation (GraphRAG)** system. Based on the architectures presented in recent legal and regulatory reasoning papers, this project demonstrates how to connect scattered corporate documents across different entities (e.g., Company X, Supplier B, Federal Regulations) to solve complex, multi-hop reasoning tasks that standard vector-based RAG cannot resolve.

## 🚀 Key Features

*   **Dual-Layer Graph Store**: Implements a structured **Knowledge Graph Skeleton** (macro-level domains/topics) and a **Text-Keyword Bipartite Graph** (micro-level entity-to-chunk and entity-to-entity RDF triplets) using `networkx`.
*   **Dynamic RDF Triplet Conversion**: Automatically extracts Subject-Predicate-Object triplets from unstructured text and JSON documents.
*   **Hybrid Predicate Categorization**: Employs a supervised classification check for statutory relations, and an unsupervised vector similarity clustering (fallback) for bespoke contractual clauses.
*   **Multi-Agent Collaborative Navigation**:
    *   **Agent 1 (The Mapper)**: Maps natural language queries to entry points on the high-level Skeleton Graph.
    *   **Agent 2 (The Navigator)**: Performs graph-aware traversal and multi-hop pathfinding to connect distributed documents.
    *   **Agent 3 (The Synthesizer)**: Compiles the facts using a Map-Reduce summary pattern and generates grounded, hallucination-free answers.
*   **Visual Debugging Dashboard**: Built on `streamlit` and `pyvis` to visually render the entities, document chunks, and the Navigator agent's active traversal path in real time.
*   **Robust Mock Fallback (Offline Mode)**: Auto-detects network errors or rate limits and falls back to a deterministic local math/regex semantic engine, ensuring the system runs immediately out of the box even without active Gemini API keys.
*   **Unicode/Windows Compatibility**: Built-in support for terminal encodings, preventing crashes on emoji rendering in Windows PowerShell.

---

## 📂 Project Structure

```text
graph_rag_mini/
├── graph_rag/                  # Core package
│   ├── agents/                 # Multi-Agent workflows
│   │   ├── __init__.py
│   │   ├── base.py             # Base agent logging & trace
│   │   ├── mapper.py           # Agent 1 (Query router)
│   │   ├── navigator.py        # Agent 2 (Graph traverser)
│   │   └── synthesizer.py      # Agent 3 (Map-Reduce compiler)
│   ├── __init__.py
│   ├── config.py               # Environmental configuration
│   ├── extractor.py            # RDF triplet & entity extractor
│   ├── graph_store.py          # Dual-layer NetworkX graph engine
│   ├── llm.py                  # Gemini API client & local mock fallbacks
│   ├── orchestrator.py         # Multi-agent workflow manager
│   └── vector_store.py         # In-memory NumPy cosine similarity vector database
├── data/
│   ├── sample_data/            # Sample files for multi-hop verification
│   │   ├── company_x_policy.json
│   │   ├── supplier_b_policy.json
│   │   └── federal_privacy_regulation.txt
│   └── indices/                # Saved JSON files for vector & graph indices
├── app.py                      # Streamlit interactive web application
├── cli.py                      # Command Line Interface (CLI)
├── run.py                      # Universal dependency manager & launcher
├── requirements.txt            # Project dependencies
├── .env.example                # Configuration template
├── README.md                   # Quickstart guide
└── PROJECT_EXPLANATION.md      # Comprehensive system architecture manual
```

---

## 🛠️ Setup Instructions

### 1. Prerequisites
Make sure Python 3.9+ is installed (compatible up to Python 3.14).

### 2. Configure Environment Variables
Copy `.env.example` to a new file named `.env`:
```bash
copy .env.example .env
```
Open `.env` and fill in your Gemini API key:
```env
GEMINI_API_KEY=your_actual_api_key_here
USE_MOCK_FALLBACK=False
```
*(Note: If you do not have an API key, keep `USE_MOCK_FALLBACK=True` to run the system completely offline in Mock Mode).*

---

## 🎮 How to Run

Use `run.py` as the universal launcher. It automatically verifies dependencies and installs missing ones on launch.

### Run via Web Interface (Recommended)
Launch the Streamlit web dashboard:
```bash
python run.py web
```
This starts a web server. Open the URL (typically `http://localhost:8501`) in your browser to:
1. Enter query searches and see the answers side-by-side with agent execution logs.
2. View the interactive knowledge graph and trace the Navigator agent's path.
3. Ingest your own TXT or JSON files dynamically.

### Run via Command-Line Interface (CLI)
You can also run index and query commands directly in your terminal:

**1. Index the Sample Corporate Documents:**
```bash
python run.py cli index
```

**2. Query the Multi-Agent System (with thinking logs):**
```bash
python run.py cli query "Does Company X's arrangement with Supplier B comply with Federal Privacy Regulations?" --verbose
```

---

## 🤖 The Multi-Hop Compliance Example

The database includes three pre-configured documents:
1.  `company_x_policy.json`: Company X partners with **Supplier B** and shares PII (Customer Data).
2.  `supplier_b_policy.json`: Supplier B processes data for Company X but stores it in **Jurisdiction F** (a foreign country).
3.  `federal_privacy_regulation.txt`: **Act Z** prohibits storing protected customer data (PII) in **Foreign Jurisdictions** without data-adequacy treaties.

**Traditional Vector RAG** fails this query because there is no direct text chunk connecting "Company X" with "Federal Privacy Regulation". It fetches disjointed chunks.
**Mini-Graph-RAG** maps the relations. The Navigator follows:
`Company X` ➔ `Supplier B` ➔ `Foreign Jurisdiction` ➔ `Federal Privacy Regulation`
and successfully synthesizes that Company X is in breach of federal guidelines.
