# Enterprise University Academic GraphRAG — SUHWAN

> **Domain:** University Academic Affairs | **Institution:** Busan University of Foreign Studies (BUFS)

An enterprise-grade Hybrid RAG chatbot for answering academic-related questions about BUFS — built with a local LLM (Ollama) and a hybrid retrieval pipeline combining Vector DB + Knowledge Graph.

---

## Hardware & Environment

| Item | Spec |
|------|------|
| OS | Windows 11 Pro (10.0.26200) |
| CPU | Intel Core i5-13600KF |
| GPU | NVIDIA GeForce RTX 4070 Ti |
| RAM | 32 GB DDR5 |
| Python | 3.12 |
| Acceleration | CUDA (GPU) |

---

## Dataset

**Source:** BUFS Official Academic Affairs PDF Documents

| Document Type | Description |
|---------------|-------------|
| Academic Handbook | Full university academic regulations (domestic/foreign/transfer students) |
| Class Schedule | Semester timetable — department/course/time/room mapping |
| Academic Calendar | Key dates: registration, exams, holidays, etc. |
| Graduation Requirements | Per-department/major credit requirements |

> All documents are parsed locally from official university PDFs. No external dataset API is used.

**Vector DB Stats (2026-1 semester):**
- Collection: `bufs_academic`
- Embedding model: `BAAI/bge-m3` (1024-dim, multilingual)
- Chunk count: ~3,000+ chunks across all document types

---

## System Architecture

```
User Query
    │
    ▼
QueryAnalyzer       → Intent classification + Entity extraction + Student ID parsing
    │
    ├──▶ ChromaDB (Vector)   → BGE-M3 embedding + top 15–20 candidates
    │        │                  (Timetable: department filter applied)
    │        └──▶ Reranker   → BGE-Reranker-v2-m3 → top 5 selected
    │
    └──▶ AcademicGraph (NetworkX) → Academic calendar, graduation requirements, dept. structure
                │
 ContextMerger ◀──┘  → Merge Vector + Graph results (token budget management)
    │
    ▼
AnswerGenerator     → Ollama (exaone3.5) LLM → final answer (streaming)
    │
    ▼
ResponseValidator   → Answer quality check
    │
    ▼
ChatLogger          → Save conversation logs as JSONL (data/logs/)
```

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| LLM | Ollama + exaone3.5:7.8b (local, GPU streaming) |
| Embedding | BAAI/bge-m3 (multilingual, 1024-dim, CPU) |
| Reranker | BAAI/bge-reranker-v2-m3 (Cross-Encoder, CPU) |
| Vector DB | ChromaDB (SQLite-based local file) |
| Knowledge Graph | NetworkX (pkl file) |
| PDF Parsing | PyMuPDF + pdfplumber |
| Timetable Parsing | Custom parser (period→time conversion, dept. normalization) |
| UI | Streamlit (multi-page: chatbot + log viewer + admin) |

---

## Project Structure

```
Enterprise-University-Academic-GraphRAG-SUHWAN/
├── app/
│   ├── config.py               # Environment variables & settings
│   ├── models.py               # Pydantic data models
│   ├── embedding/
│   │   └── embedder.py         # BGE-M3 embedding wrapper
│   ├── pipeline/
│   │   ├── query_analyzer.py   # Intent classification + entity extraction
│   │   ├── query_router.py     # Vector/Graph routing (dept. filter)
│   │   ├── reranker.py         # BGE-Reranker-v2-m3 reranking
│   │   ├── context_merger.py   # Search result merging (token budget)
│   │   ├── answer_generator.py # LLM answer generation
│   │   ├── glossary.py         # Domain term normalization (Korean aliases)
│   │   └── response_validator.py
│   ├── vectordb/
│   │   └── chroma_store.py     # ChromaDB CRUD (dept. filter support)
│   ├── graphdb/
│   │   └── academic_graph.py   # NetworkX knowledge graph
│   ├── pdf/
│   │   ├── detector.py         # PDF type auto-detection
│   │   ├── digital_extractor.py# Digital PDF parsing + chunking
│   │   ├── ocr_extractor.py    # Scanned PDF OCR
│   │   └── timetable_parser.py # Class schedule dedicated parser
│   ├── logging/
│   │   └── chat_logger.py      # Conversation log (JSONL, data/logs/)
│   └── ui/
│       └── chat_app.py         # Streamlit chat UI
├── pages/
│   ├── admin.py                # Admin page (graduation requirements mgmt)
│   └── logs.py                 # Log viewer page
├── scripts/
│   ├── ingest_pdf.py           # PDF → ChromaDB ingestion
│   ├── build_graph.py          # Knowledge graph build
│   ├── pdf_to_graph.py         # Auto-parse academic data from PDF
│   ├── make_eval_dataset.py    # Generate evaluation JSONL
│   ├── evaluate.py             # Auto evaluation + LLM-as-a-Judge
│   ├── qualitative_judge.py    # Qualitative judge (0–5 scale)
│   ├── compare_eval.py         # Compare evaluation results
│   └── make_report.py          # Evaluation report generation
├── data/
│   ├── pdfs/                   # Academic PDF files (gitignored)
│   ├── chromadb/               # ChromaDB persistent store (gitignored)
│   ├── graphs/                 # academic_graph.pkl (gitignored)
│   ├── logs/                   # Chat logs JSONL (gitignored)
│   └── eval/                   # Evaluation datasets & results
├── tests/                      # 52 unit tests
├── main.py                     # Streamlit entry point
├── requirements.txt
└── .env.example                # Environment variable template
```

---

## Installation

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- ~4 GB+ RAM (BGE-M3 model)
- NVIDIA GPU recommended (CUDA) for LLM inference

### 1. Clone & Set Up Virtual Environment

```bash
git clone https://github.com/Enterprise-GraphRAG-Study/Enterprise-University-Academic-GraphRAG-SUHWAN.git
cd Enterprise-University-Academic-GraphRAG-SUHWAN

# Create virtual environment
python -m venv venv

# Activate (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Activate (Mac/Linux)
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Pull Ollama Model

```bash
ollama pull exaone3.5:7.8b
# For low-spec environments
ollama pull exaone3.5:2.4b
```

### 4. Configure Environment Variables

```bash
cp .env.example .env
# Edit .env with your settings
```

---

## Data Preparation

### PDF Ingestion (Vector DB)

```bash
# Ingest academic handbook PDF
python scripts/ingest_pdf.py --pdf data/pdfs/academic_handbook.pdf --doc-type domestic

# Ingest class schedule PDF
python scripts/ingest_pdf.py --pdf data/pdfs/timetable.pdf --doc-type timetable --semester 2026-1

# Ingest entire directory
python scripts/ingest_pdf.py --dir data/pdfs/

# Check DB status
python scripts/ingest_pdf.py --status
```

`--doc-type` options: `domestic` / `foreign` / `transfer` / `schedule` / `timetable`

### Knowledge Graph Build

```bash
# Auto-parse academic data from PDF and build graph
python scripts/pdf_to_graph.py --pdf data/pdfs/academic_handbook.pdf

# Dry run (check parse results without changes)
python scripts/pdf_to_graph.py --pdf data/pdfs/academic_handbook.pdf --dry-run
```

---

## Run

```bash
streamlit run main.py
```

| Page | URL | Description |
|------|-----|-------------|
| Chatbot (main) | `http://localhost:8501` | Academic Q&A |
| Log Viewer | `http://localhost:8501/logs` | Browse & download conversation logs (CSV) |
| Admin | `http://localhost:8501/admin` | Manage graduation requirements |

---

## Key Features

| Feature | Description |
|---------|-------------|
| Hybrid Retrieval | Vector (ChromaDB) + Graph (NetworkX) result merging |
| Dept. Timetable Filter | `department` filter prevents cross-dept. chunk contamination |
| Star Feedback | 1–5 rating per answer, saved in logs |
| Chat Logging | Daily JSONL files, downloadable as CSV from UI |
| Loading Animation | Book-flipping animation during answer generation |
| Student ID Recognition | Auto-parses student ID from query → personalized graduation info |
| Domain Term Normalization | Korean abbreviations/aliases → official term mapping |
| Portal Links | Quick-access panel for academic portals |
| Admin Panel | Add/edit/delete graduation requirements per dept./major |

---

## Evaluation Results

> **Dataset:** 2026-1 semester, 50 questions | **Date:** 2026-03-09

| Metric | Value |
|--------|-------|
| Hit Rate | **100.0%** |
| Contains GT | 74.0% |
| LLM Judge Correctness | **94.0%** (47/50) |
| Qualitative Judge (avg accuracy) | **4.8 / 5** |
| Answer Success Rate | 100.0% |
| Source Citation Rate | 100.0% |
| Avg. Retrieval Time | ~17 sec |
| Avg. Generation Time | ~1.8 sec |

**Accuracy by Difficulty:**

| Difficulty | Accuracy | n |
|------------|----------|---|
| Easy | 4.77 / 5 | 31 |
| Medium | 4.81 / 5 | 16 |
| Hard | 5.00 / 5 | 3 |

**Error Analysis (3 cases):**
- `incomplete` (3 cases): Partial answer — answer included correct info but missed edge conditions

---

## Tests

```bash
# Activate venv first
.\venv\Scripts\Activate.ps1

# Run all 52 tests
pytest tests/ -v
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `exaone3.5:7.8b` | Main LLM model |
| `OLLAMA_FALLBACK_MODEL` | `exaone3.5:2.4b` | Fallback model |
| `OLLAMA_NUM_CTX` | `2048` | Context length |
| `OLLAMA_TEMPERATURE` | `0.1` | Generation temperature |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | Embedding model |
| `EMBEDDING_DEVICE` | `cpu` | Embedding device |
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | Reranker model |
| `RERANKER_DEVICE` | `cpu` | Reranker device |
| `RERANKER_ENABLED` | `true` | Enable reranking |
| `RERANKER_TOP_K` | `5` | Top-k after reranking |
| `RERANKER_CANDIDATE_K` | `15` | Candidates before reranking |
| `CHROMA_N_RESULTS` | `15` | ChromaDB search results count |
| `CHROMA_COLLECTION` | `bufs_academic` | Collection name |

---

## Roadmap

- [x] **Week 1:** Domain selection, environment setup, PDF ingestion pipeline
- [x] **Week 2:** Knowledge graph construction (NetworkX), hybrid retrieval
- [x] **Week 3:** Reranker integration (BGE-Reranker-v2-m3), query routing
- [x] **Week 4:** Evaluation pipeline (Hit Rate, LLM Judge, Qualitative Judge)
- [x] **Week 5:** Admin panel, graduation requirement management
- [ ] **Week 6+:** GraphRAG integration (Neo4j/LangGraph migration)

---

## License

MIT License — see [LICENSE](LICENSE) for details.
