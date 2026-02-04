# Architecture Documentation

## System Overview

The Multimodal Medical RAG system is designed to enable intelligent querying of medical imaging data and clinical documents using a Retrieval-Augmented Generation (RAG) approach.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA INGESTION LAYER                         │
├─────────────────────────────────────────────────────────────────────┤
│  Medical Images    Clinical Reports    Structured Data              │
│  (DICOM, PNG)      (PDF, TXT)         (CSV, JSON)                  │
│       │                 │                  │                        │
│       └────────────────┬┴──────────────────┘                        │
│                        ▼                                            │
│              ┌─────────────────┐                                    │
│              │  Cloud Storage  │ ◄── Raw data buckets              │
│              │     (GCS)       │                                    │
│              └────────┬────────┘                                    │
└───────────────────────┼─────────────────────────────────────────────┘
                        │
┌───────────────────────┼─────────────────────────────────────────────┐
│                       ▼          EMBEDDING LAYER                     │
├─────────────────────────────────────────────────────────────────────┤
│              ┌─────────────────┐                                    │
│              │   Vertex AI     │                                    │
│              │   Embeddings    │                                    │
│              └────────┬────────┘                                    │
│                       │                                             │
│         ┌─────────────┼─────────────┐                               │
│         ▼             ▼             ▼                               │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐                      │
│  │   Text     │ │  Image     │ │ Multimodal │                      │
│  │ Embedding  │ │ Embedding  │ │ Embedding  │                      │
│  │ (768-dim)  │ │(1408-dim)  │ │ (combined) │                      │
│  └─────┬──────┘ └─────┬──────┘ └─────┬──────┘                      │
│        └──────────────┼──────────────┘                              │
│                       ▼                                             │
│              ┌─────────────────┐                                    │
│              │  Embeddings     │ ◄── Processed embeddings          │
│              │  Storage (GCS)  │                                    │
│              └────────┬────────┘                                    │
└───────────────────────┼─────────────────────────────────────────────┘
                        │
┌───────────────────────┼─────────────────────────────────────────────┐
│                       ▼         RETRIEVAL LAYER                      │
├─────────────────────────────────────────────────────────────────────┤
│              ┌─────────────────┐                                    │
│              │  Vector Search  │ ◄── Matching Engine               │
│              │   (Vertex AI)   │     (ScaNN algorithm)             │
│              └────────┬────────┘                                    │
│                       │                                             │
│              ┌────────┴────────┐                                    │
│              ▼                 ▼                                    │
│     ┌─────────────┐   ┌─────────────┐                              │
│     │   K-NN      │   │  Metadata   │                              │
│     │   Search    │   │  Filtering  │                              │
│     └──────┬──────┘   └──────┬──────┘                              │
│            └─────────┬───────┘                                      │
│                      ▼                                              │
│           ┌──────────────────┐                                      │
│           │ Retrieved Context│                                      │
│           └────────┬─────────┘                                      │
└────────────────────┼────────────────────────────────────────────────┘
                     │
┌────────────────────┼────────────────────────────────────────────────┐
│                    ▼         GENERATION LAYER                        │
├─────────────────────────────────────────────────────────────────────┤
│           ┌──────────────────┐                                      │
│           │  Context Builder │ ◄── Assemble retrieved docs         │
│           └────────┬─────────┘                                      │
│                    ▼                                                │
│           ┌──────────────────┐                                      │
│           │   Gemini LLM     │ ◄── Context-aware generation        │
│           │   (Vertex AI)    │                                      │
│           └────────┬─────────┘                                      │
│                    ▼                                                │
│           ┌──────────────────┐                                      │
│           │    Response      │ ◄── Answer + citations              │
│           └──────────────────┘                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Component Details

### Data Ingestion

| Component | Technology | Purpose |
|-----------|------------|---------|
| Image Processing | PyDICOM, PIL | Convert medical images to standard format |
| Text Extraction | Python PDF libraries | Extract text from clinical documents |
| Data Validation | Pydantic | Validate data integrity |
| Storage | Cloud Storage | Durable, scalable object storage |

### Embedding Generation

| Model | Dimensions | Use Case |
|-------|------------|----------|
| text-embedding-004 | 768 | Clinical text, reports |
| multimodalembedding@001 | 1408 | Medical images |

### Vector Search

- **Algorithm**: ScaNN (Scalable Nearest Neighbors)
- **Distance Metric**: Dot Product
- **Approximate Neighbors**: 150
- **Query Latency**: <100ms at scale

### Generation

- **Model**: Gemini 1.5 Pro
- **Context Window**: 8,000 tokens
- **Temperature**: 0.2 (low for factual accuracy)

## Data Flow

1. **Ingestion**: Raw data uploaded to GCS
2. **Processing**: Data cleaned and validated
3. **Embedding**: Vectors generated via Vertex AI
4. **Indexing**: Embeddings added to Vector Search
5. **Query**: User query embedded and searched
6. **Retrieval**: Top-K relevant documents retrieved
7. **Generation**: LLM generates response with context
8. **Response**: Answer returned with citations

## Security Considerations

- All data at rest encrypted with Google-managed keys
- IAM roles follow principle of least privilege
- No PHI/PII in synthetic development data
- Production data requires HIPAA BAA

## Scalability

| Component | Scaling Strategy |
|-----------|------------------|
| Storage | Automatic (GCS) |
| Embeddings | Batch processing, parallel workers |
| Vector Search | Sharding, replica endpoints |
| LLM | Request-based, auto-scaling |
