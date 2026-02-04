#!/usr/bin/env python3
# ============================================
# Architecture Diagrams for Multimodal Medical RAG
# ============================================
"""
This script generates architecture diagrams for the Multimodal Medical RAG system.

Diagrams Generated:
    1. Simple RAG Architecture - Basic RAG pattern for comparison
    2. Agentic RAG Architecture - Our advanced multi-agent system
    3. Data Pipeline Architecture - End-to-end data flow

Requirements:
    - Graphviz installed (choco install graphviz / brew install graphviz)
    - diagrams library (pip install diagrams)

Usage:
    python docs/architecture_diagrams.py

Output:
    PNG files in docs/diagrams/
"""

import os
from pathlib import Path

# ============================================
# Diagrams Library Imports
# ============================================
from diagrams import Diagram, Cluster, Edge

# GCP-specific node imports
from diagrams.gcp.compute import Run, Functions
from diagrams.gcp.analytics import BigQuery
from diagrams.gcp.storage import GCS
from diagrams.gcp.ml import AIHub, VisionAPI, NaturalLanguageAPI, AIPlatform
from diagrams.gcp.network import LoadBalancing
from diagrams.gcp.api import APIGateway

# Generic nodes for custom components
from diagrams.generic.compute import Rack
from diagrams.generic.storage import Storage
from diagrams.generic.database import SQL

# On-premise/custom nodes
from diagrams.onprem.client import User, Users
from diagrams.onprem.compute import Server

# Programming nodes
from diagrams.programming.flowchart import Decision, Action


# ============================================
# Configuration
# ============================================

# Output directory for generated diagrams
OUTPUT_DIR = Path(__file__).parent / "diagrams"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Common diagram attributes for consistent styling
DIAGRAM_ATTRS = {
    "fontsize": "14",
    "fontname": "Helvetica",
    "bgcolor": "white",
    "pad": "0.5",
}

# Node attributes for professional appearance
NODE_ATTRS = {
    "fontsize": "11",
    "fontname": "Helvetica",
}

# Edge attributes
EDGE_ATTRS = {
    "fontsize": "10",
    "fontname": "Helvetica",
}


# ============================================
# DIAGRAM 1: Simple RAG Architecture
# ============================================
def create_simple_rag_diagram():
    """
    Creates a diagram showing a basic RAG (Retrieval-Augmented Generation) architecture.

    This represents a traditional RAG pattern where:
    - User sends a query
    - System retrieves relevant context from vector store
    - LLM generates response using the context

    This is included for COMPARISON with our more advanced Agentic RAG system.
    """

    with Diagram(
        "Simple RAG Architecture",
        filename=str(OUTPUT_DIR / "01_simple_rag"),
        show=False,  # Don't auto-open the image
        direction="LR",  # Left to Right flow
        graph_attr=DIAGRAM_ATTRS,
        node_attr=NODE_ATTRS,
        edge_attr=EDGE_ATTRS,
    ):
        # ----------------------------------------
        # USER LAYER
        # The entry point - users interact via API
        # ----------------------------------------
        user = User("Medical\nProfessional")

        # ----------------------------------------
        # API LAYER
        # API Gateway handles authentication, rate limiting
        # Cloud Run hosts the main application
        # ----------------------------------------
        with Cluster("API Layer"):
            api_gateway = APIGateway("API Gateway")
            cloud_run = Run("Cloud Run\n(RAG Service)")

        # ----------------------------------------
        # AI/ML LAYER
        # Vertex AI provides both:
        # - Gemini for text generation
        # - Vector Search for similarity matching
        # ----------------------------------------
        with Cluster("Vertex AI"):
            # Gemini LLM for response generation
            # Uses retrieved context to generate accurate answers
            gemini = AIPlatform("Gemini\n(Generation)")

            # Vector Search for semantic similarity
            # Finds relevant documents based on query embedding
            vector_search = AIPlatform("Vector Search\n(Retrieval)")

        # ----------------------------------------
        # DATA LAYER
        # Where all the medical data is stored:
        # - Cloud Storage: Images, PDFs, raw files
        # - BigQuery: Structured metadata, query logs
        # ----------------------------------------
        with Cluster("Data Storage"):
            # Cloud Storage for unstructured data
            # Medical images (DICOM, PNG), clinical reports (PDF)
            gcs = GCS("Cloud Storage\n(Images/Docs)")

            # BigQuery for structured data
            # Patient metadata, document indices, analytics
            bq = BigQuery("BigQuery\n(Metadata)")

        # ----------------------------------------
        # DATA FLOW CONNECTIONS
        # Showing how data moves through the system
        # ----------------------------------------

        # User → API Gateway → Cloud Run
        # Standard request flow through the API layer
        user >> Edge(label="Query") >> api_gateway >> cloud_run

        # Cloud Run → Vertex AI services
        # The RAG service coordinates between retrieval and generation
        cloud_run >> Edge(label="1. Embed\nQuery") >> vector_search
        cloud_run >> Edge(label="3. Generate\nResponse") >> gemini

        # Cloud Run → Data Storage
        # Fetching raw data and metadata as needed
        cloud_run >> Edge(label="2. Fetch\nContext") >> gcs
        cloud_run >> Edge(label="Structured\nQueries") >> bq


# ============================================
# DIAGRAM 2: Agentic RAG Architecture
# ============================================
def create_agentic_rag_diagram():
    """
    Creates a diagram showing our advanced Agentic RAG architecture.

    This is our MAIN SYSTEM architecture featuring:
    - Orchestrator Agent: Coordinates the overall workflow
    - Planning Agent: Determines optimal query strategy
    - Multiple Tools: Specialized retrieval and computation
    - Specialist Agents: Domain-specific processing

    Key advantages over Simple RAG:
    - Multi-step reasoning
    - Tool selection based on query type
    - Specialized agents for complex tasks
    - Better handling of multimodal data
    """

    with Diagram(
        "Agentic RAG Architecture - Multimodal Medical System",
        filename=str(OUTPUT_DIR / "02_agentic_rag"),
        show=False,
        direction="TB",  # Top to Bottom for agent hierarchy
        graph_attr={**DIAGRAM_ATTRS, "ranksep": "1.0", "nodesep": "0.5"},
        node_attr=NODE_ATTRS,
        edge_attr=EDGE_ATTRS,
    ):
        # ----------------------------------------
        # USER INTERFACE LAYER
        # Entry point for medical professionals
        # ----------------------------------------
        users = Users("Medical Staff\n& Researchers")

        # ----------------------------------------
        # API & ORCHESTRATION LAYER
        # The "brain" of the system
        # ----------------------------------------
        with Cluster("Orchestration Layer"):
            api = APIGateway("API Gateway\n(Auth/Rate Limit)")

            # Main orchestrator - coordinates all agents and tools
            # Implements ReAct pattern (Reason + Act)
            orchestrator = Run("Orchestrator Agent\n(Cloud Run)")

            # Planning agent - analyzes query and creates execution plan
            # Determines which tools/agents to use
            planner = Run("Planning Agent\n(Query Analysis)")

        # ----------------------------------------
        # TOOLS LAYER
        # Specialized tools the orchestrator can invoke
        # Each tool has a specific purpose
        # ----------------------------------------
        with Cluster("Tools (Function Calling)"):
            # Text Search Tool
            # Retrieves relevant text documents using embeddings
            text_tool = Action("Text Search\nTool")

            # Image Search Tool
            # Retrieves similar medical images using visual embeddings
            image_tool = Action("Image Search\nTool")

            # SQL Query Tool
            # Executes structured queries against BigQuery
            sql_tool = Action("SQL Query\nTool")

            # Calculator Tool
            # Performs medical calculations (dosage, BMI, etc.)
            calc_tool = Action("Calculator\nTool")

        # ----------------------------------------
        # SPECIALIST AGENTS LAYER
        # Domain-specific agents for complex tasks
        # ----------------------------------------
        with Cluster("Specialist Agents"):
            # Image Analysis Agent
            # Uses Vertex AI Vision for medical image interpretation
            # Detects abnormalities, generates descriptions
            image_agent = Run("Image Analysis\nAgent")

            # Report Generator Agent
            # Creates structured medical reports from findings
            # Uses Gemini for natural language generation
            report_agent = Run("Report Generator\nAgent")

        # ----------------------------------------
        # VERTEX AI SERVICES
        # Core AI/ML capabilities
        # ----------------------------------------
        with Cluster("Vertex AI Services"):
            # Vector Search - semantic similarity matching
            vector_search = AIPlatform("Vector Search\n(Embeddings)")

            # Vision API - medical image analysis
            vision = VisionAPI("Vision API\n(Image Analysis)")

            # Gemini - text generation and reasoning
            gemini = AIPlatform("Gemini Pro\n(Generation)")

        # ----------------------------------------
        # DATA LAYER
        # Centralized data storage
        # ----------------------------------------
        with Cluster("Data Storage"):
            # Cloud Storage - all raw and processed data
            gcs = GCS("Cloud Storage\n(Medical Data)")

            # BigQuery - structured metadata and analytics
            bq = BigQuery("BigQuery\n(Metadata)")

        # ----------------------------------------
        # CONNECTION FLOW
        # ----------------------------------------

        # User → API → Orchestrator flow
        users >> api >> orchestrator

        # Orchestrator → Planning (first step in every request)
        orchestrator >> Edge(label="1. Plan") >> planner

        # Orchestrator → Tools (based on plan)
        orchestrator >> Edge(label="2. Execute") >> text_tool
        orchestrator >> Edge(label="2. Execute") >> image_tool
        orchestrator >> Edge(label="2. Execute") >> sql_tool
        orchestrator >> Edge(label="2. Execute") >> calc_tool

        # Orchestrator → Specialist Agents (for complex tasks)
        orchestrator >> Edge(label="3. Specialize") >> image_agent
        orchestrator >> Edge(label="3. Specialize") >> report_agent

        # Tools → Vertex AI services
        text_tool >> vector_search
        image_tool >> vector_search

        # Tools → Data storage
        sql_tool >> bq

        # Specialist Agents → Vertex AI
        image_agent >> vision
        report_agent >> gemini

        # All data flows through Cloud Storage
        vector_search >> Edge(style="dashed") >> gcs
        bq >> Edge(style="dashed") >> gcs


# ============================================
# DIAGRAM 3: Data Pipeline Architecture
# ============================================
def create_data_pipeline_diagram():
    """
    Creates a diagram showing the data pipeline architecture.

    This shows how data flows from raw ingestion to queryable embeddings:

    Bronze Layer (Raw):
        - Original medical images and documents
        - No transformations applied

    Silver Layer (Processed):
        - Cleaned and validated data
        - Extracted text, normalized images

    Gold Layer (Embeddings):
        - Vector embeddings for similarity search
        - Indexed in Vector Search

    Metadata:
        - Structured information in BigQuery
        - Enables hybrid search (vector + filters)
    """

    with Diagram(
        "Data Pipeline Architecture - ETL & Embedding Generation",
        filename=str(OUTPUT_DIR / "03_data_pipeline"),
        show=False,
        direction="LR",  # Left to Right for pipeline flow
        graph_attr={**DIAGRAM_ATTRS, "ranksep": "1.2"},
        node_attr=NODE_ATTRS,
        edge_attr=EDGE_ATTRS,
    ):
        # ----------------------------------------
        # DATA SOURCES
        # Where raw medical data originates
        # ----------------------------------------
        with Cluster("Data Sources"):
            # DICOM images from medical imaging devices
            dicom_source = Storage("DICOM\nImages")

            # Clinical reports in various formats
            reports_source = Storage("Clinical\nReports")

            # Structured patient data
            structured_source = SQL("Structured\nData")

        # ----------------------------------------
        # BRONZE LAYER (RAW)
        # Landing zone for all incoming data
        # No transformations - preserve original state
        # ----------------------------------------
        with Cluster("Bronze Layer (Raw)"):
            # Raw data bucket - immutable storage
            bronze_bucket = GCS("GCS: Raw Data\n(Immutable)")

            # Cloud Function triggered on new uploads
            # Validates and routes data to processing
            ingest_function = Functions("Ingestion\nTrigger")

        # ----------------------------------------
        # PROCESSING LAYER
        # Transforms raw data into usable formats
        # ----------------------------------------
        with Cluster("Processing Layer"):
            # Main processing service
            # Handles: image conversion, text extraction, validation
            processor = Run("Data Processor\n(Cloud Run)")

            # Parallel processing tasks
            with Cluster("Processing Tasks"):
                # Image processing: DICOM → PNG, normalization
                img_process = Action("Image\nProcessing")

                # Text extraction: OCR, PDF parsing
                text_process = Action("Text\nExtraction")

                # Validation: schema checks, quality filters
                validation = Action("Data\nValidation")

        # ----------------------------------------
        # SILVER LAYER (PROCESSED)
        # Clean, validated, standardized data
        # ----------------------------------------
        with Cluster("Silver Layer (Processed)"):
            # Processed data bucket
            silver_bucket = GCS("GCS: Processed\n(Standardized)")

        # ----------------------------------------
        # EMBEDDING GENERATION
        # Creates vector representations for search
        # ----------------------------------------
        with Cluster("Embedding Generation"):
            # Embedding service - orchestrates embedding creation
            embed_service = Run("Embedding\nService")

            # Vertex AI embedding models
            with Cluster("Vertex AI"):
                # Text embeddings (768 dimensions)
                text_embed = AIPlatform("Text Embeddings\n(768-dim)")

                # Multimodal embeddings (1408 dimensions)
                image_embed = AIPlatform("Image Embeddings\n(1408-dim)")

        # ----------------------------------------
        # GOLD LAYER (EMBEDDINGS)
        # Query-ready vector embeddings
        # ----------------------------------------
        with Cluster("Gold Layer (Embeddings)"):
            # Embedding storage
            gold_bucket = GCS("GCS: Embeddings\n(Vectors)")

            # Vector Search index for fast retrieval
            vector_index = AIPlatform("Vector Search\nIndex")

        # ----------------------------------------
        # METADATA LAYER
        # Structured information for filtering
        # ----------------------------------------
        with Cluster("Metadata Layer"):
            # BigQuery tables
            bq = BigQuery("BigQuery\n(Metadata)")

            # Document registry
            doc_registry = SQL("Document\nRegistry")

        # ----------------------------------------
        # PIPELINE FLOW CONNECTIONS
        # ----------------------------------------

        # Sources → Bronze (ingestion)
        dicom_source >> Edge(label="Upload") >> bronze_bucket
        reports_source >> Edge(label="Upload") >> bronze_bucket
        structured_source >> Edge(label="Upload") >> bronze_bucket

        # Bronze → Trigger → Processing
        bronze_bucket >> ingest_function >> processor

        # Processor → Processing Tasks
        processor >> img_process
        processor >> text_process
        processor >> validation

        # Processing → Silver
        img_process >> silver_bucket
        text_process >> silver_bucket
        validation >> silver_bucket

        # Silver → Embedding Service
        silver_bucket >> embed_service

        # Embedding Service → Vertex AI
        embed_service >> text_embed
        embed_service >> image_embed

        # Vertex AI → Gold Layer
        text_embed >> gold_bucket
        image_embed >> gold_bucket

        # Gold → Vector Index
        gold_bucket >> Edge(label="Index") >> vector_index

        # Metadata flow (parallel to main pipeline)
        processor >> Edge(style="dashed", label="Metadata") >> bq
        bq >> doc_registry


# ============================================
# MAIN EXECUTION
# ============================================
def main():
    """
    Generate all architecture diagrams.

    Each diagram is saved as a PNG file in docs/diagrams/
    """
    print("=" * 60)
    print("Generating Architecture Diagrams")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    # Generate each diagram
    diagrams = [
        ("Simple RAG Architecture", create_simple_rag_diagram),
        ("Agentic RAG Architecture", create_agentic_rag_diagram),
        ("Data Pipeline Architecture", create_data_pipeline_diagram),
    ]

    for name, create_func in diagrams:
        print(f"Creating: {name}...")
        try:
            create_func()
            print(f"  [OK] Generated successfully")
        except Exception as e:
            print(f"  [ERROR] {e}")

    print()
    print("=" * 60)
    print("All diagrams generated!")
    print(f"View them in: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
