# Architecture Diagrams

This folder contains architecture diagrams for the Multimodal Medical RAG system.

## Diagrams Overview

### 1. Simple RAG Architecture (`01_simple_rag.png`)

A basic Retrieval-Augmented Generation pattern showing:

- **User Layer**: Medical professionals accessing the system
- **API Layer**: API Gateway + Cloud Run service
- **AI Layer**: Vertex AI Gemini (generation) and Vector Search (retrieval)
- **Data Layer**: Cloud Storage (images/docs) and BigQuery (metadata)

This diagram serves as a **baseline comparison** for understanding how our more advanced Agentic RAG system differs from traditional RAG implementations.

**Flow**: User query → API Gateway → Cloud Run → Vector Search (retrieve context) → Gemini (generate response)

---

### 2. Agentic RAG Architecture (`02_agentic_rag.png`)

Our **main system architecture** featuring a multi-agent approach:

- **Orchestration Layer**:
  - Orchestrator Agent: Coordinates workflow using ReAct pattern
  - Planning Agent: Analyzes queries and creates execution plans

- **Tools Layer** (Function Calling):
  - Text Search Tool → Vector Search
  - Image Search Tool → Vector Search
  - SQL Query Tool → BigQuery
  - Calculator Tool (dosage, BMI calculations)

- **Specialist Agents**:
  - Image Analysis Agent → Vision API
  - Report Generator Agent → Gemini

- **Vertex AI Services**: Vector Search, Vision API, Gemini Pro

- **Data Storage**: Cloud Storage + BigQuery

**Key Advantages over Simple RAG**:
- Multi-step reasoning capabilities
- Dynamic tool selection based on query type
- Specialized agents for complex medical tasks
- Superior handling of multimodal data (text + images)

---

### 3. Data Pipeline Architecture (`03_data_pipeline.png`)

End-to-end data flow from ingestion to queryable embeddings:

- **Bronze Layer (Raw)**: Landing zone for DICOM images, clinical reports, structured data
- **Processing Layer**: Cloud Run processor with image processing, text extraction, validation
- **Silver Layer (Processed)**: Clean, standardized data ready for embedding
- **Embedding Generation**: Vertex AI text embeddings (768-dim) and image embeddings (1408-dim)
- **Gold Layer (Embeddings)**: Vector Search index for fast retrieval
- **Metadata Layer**: BigQuery document registry for hybrid search (vector + filters)

**Data Flow**:
```
Sources → Bronze (raw) → Processing → Silver (clean) → Embeddings → Gold (indexed) → Vector Search
                                ↓
                           BigQuery (metadata)
```

---

## Regenerating Diagrams

To regenerate these diagrams:

```bash
cd multimodal-medical-rag-gcp
python docs/architecture_diagrams.py
```

### Requirements

- Python 3.8+
- Graphviz (`choco install graphviz` on Windows)
- diagrams library (`pip install diagrams`)

## Customization

Edit `docs/architecture_diagrams.py` to:
- Modify component styling (`DIAGRAM_ATTRS`, `NODE_ATTRS`, `EDGE_ATTRS`)
- Add/remove components from diagrams
- Change diagram direction (`LR` for left-right, `TB` for top-bottom)
- Add new diagrams by creating additional `create_*_diagram()` functions
