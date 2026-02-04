# ============================================
# BigQuery Module
# ============================================
# Creates BigQuery dataset and tables for metadata storage
#
# Tables created:
#   - documents: Metadata for indexed documents
#   - query_logs: Logs of RAG queries and responses

terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

# ------------------------------
# Variables
# ------------------------------

variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "us-central1"
}

variable "environment" {
  description = "Environment name (dev/test/prod)"
  type        = string
}

variable "delete_contents_on_destroy" {
  description = "Allow deletion of non-empty dataset"
  type        = bool
  default     = false
}

variable "labels" {
  description = "Labels to apply to all resources"
  type        = map(string)
  default     = {}
}

# ------------------------------
# Local Values
# ------------------------------

locals {
  dataset_id = replace("${var.project_id}_metadata_${var.environment}", "-", "_")
}

# ------------------------------
# Dataset
# ------------------------------

resource "google_bigquery_dataset" "metadata" {
  project                    = var.project_id
  dataset_id                 = local.dataset_id
  friendly_name              = "Medical RAG Metadata (${var.environment})"
  description                = "Metadata storage for the medical RAG system"
  location                   = var.region
  delete_contents_on_destroy = var.delete_contents_on_destroy

  labels = var.labels
}

# ------------------------------
# Documents Table
# ------------------------------

resource "google_bigquery_table" "documents" {
  project    = var.project_id
  dataset_id = google_bigquery_dataset.metadata.dataset_id
  table_id   = "documents"

  description = "Metadata for indexed medical documents"

  deletion_protection = var.environment == "prod" ? true : false

  schema = jsonencode([
    {
      name        = "document_id"
      type        = "STRING"
      mode        = "REQUIRED"
      description = "Unique document identifier"
    },
    {
      name        = "source_path"
      type        = "STRING"
      mode        = "REQUIRED"
      description = "GCS path to source document"
    },
    {
      name        = "document_type"
      type        = "STRING"
      mode        = "REQUIRED"
      description = "Type: image, report, study"
    },
    {
      name        = "modality"
      type        = "STRING"
      mode        = "NULLABLE"
      description = "Imaging modality (CT, MRI, XR)"
    },
    {
      name        = "body_part"
      type        = "STRING"
      mode        = "NULLABLE"
      description = "Body part examined"
    },
    {
      name        = "embedding_path"
      type        = "STRING"
      mode        = "NULLABLE"
      description = "GCS path to embedding vector"
    },
    {
      name        = "text_content"
      type        = "STRING"
      mode        = "NULLABLE"
      description = "Extracted text content"
    },
    {
      name        = "metadata"
      type        = "JSON"
      mode        = "NULLABLE"
      description = "Additional metadata as JSON"
    },
    {
      name        = "created_at"
      type        = "TIMESTAMP"
      mode        = "REQUIRED"
      description = "Record creation timestamp"
    },
    {
      name        = "updated_at"
      type        = "TIMESTAMP"
      mode        = "REQUIRED"
      description = "Last update timestamp"
    }
  ])

  labels = merge(var.labels, {
    table-type = "documents"
  })
}

# ------------------------------
# Query Logs Table
# ------------------------------

resource "google_bigquery_table" "query_logs" {
  project    = var.project_id
  dataset_id = google_bigquery_dataset.metadata.dataset_id
  table_id   = "query_logs"

  description = "Logs of RAG queries and responses for analytics"

  deletion_protection = false

  time_partitioning {
    type  = "DAY"
    field = "timestamp"
  }

  schema = jsonencode([
    {
      name        = "query_id"
      type        = "STRING"
      mode        = "REQUIRED"
      description = "Unique query identifier"
    },
    {
      name        = "timestamp"
      type        = "TIMESTAMP"
      mode        = "REQUIRED"
      description = "Query timestamp"
    },
    {
      name        = "query_text"
      type        = "STRING"
      mode        = "REQUIRED"
      description = "User query text"
    },
    {
      name        = "response_text"
      type        = "STRING"
      mode        = "NULLABLE"
      description = "Generated response"
    },
    {
      name        = "retrieved_doc_ids"
      type        = "STRING"
      mode        = "REPEATED"
      description = "IDs of retrieved documents"
    },
    {
      name        = "confidence_score"
      type        = "FLOAT64"
      mode        = "NULLABLE"
      description = "Response confidence score"
    },
    {
      name        = "latency_ms"
      type        = "INT64"
      mode        = "NULLABLE"
      description = "Total query latency in milliseconds"
    },
    {
      name        = "user_feedback"
      type        = "STRING"
      mode        = "NULLABLE"
      description = "User feedback (positive/negative/null)"
    }
  ])

  labels = merge(var.labels, {
    table-type = "query-logs"
  })
}

# ------------------------------
# Outputs
# ------------------------------

output "dataset_id" {
  description = "BigQuery dataset ID"
  value       = google_bigquery_dataset.metadata.dataset_id
}

output "documents_table_id" {
  description = "Full ID of documents table"
  value       = "${var.project_id}.${google_bigquery_dataset.metadata.dataset_id}.${google_bigquery_table.documents.table_id}"
}

output "query_logs_table_id" {
  description = "Full ID of query logs table"
  value       = "${var.project_id}.${google_bigquery_dataset.metadata.dataset_id}.${google_bigquery_table.query_logs.table_id}"
}
