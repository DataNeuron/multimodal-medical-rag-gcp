# ============================================
# Vertex AI Module
# ============================================
# Provisions Vertex AI resources for embeddings and vector search
#
# Resources created:
#   - Vector Search Index (Matching Engine)
#   - Index Endpoint for serving
#   - Service account with appropriate permissions

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

variable "embedding_dimension" {
  description = "Dimension of embedding vectors"
  type        = number
  default     = 768
}

variable "shard_size" {
  description = "Shard size for vector index"
  type        = string
  default     = "SHARD_SIZE_SMALL"
}

variable "machine_type" {
  description = "Machine type for index endpoint"
  type        = string
  default     = "n1-standard-4"
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
  index_name          = "medical-rag-index-${var.environment}"
  endpoint_name       = "medical-rag-endpoint-${var.environment}"
  deployed_index_name = "medical-rag-deployed-${var.environment}"
}

# ------------------------------
# Vector Search Index
# ------------------------------

resource "google_vertex_ai_index" "medical_rag" {
  project      = var.project_id
  region       = var.region
  display_name = local.index_name
  description  = "Vector index for medical RAG embeddings (${var.environment})"

  metadata {
    contents_delta_uri = ""  # Will be set during data upload

    config {
      dimensions                  = var.embedding_dimension
      approximate_neighbors_count = 150
      shard_size                 = var.shard_size

      distance_measure_type = "DOT_PRODUCT_DISTANCE"

      algorithm_config {
        tree_ah_config {
          leaf_node_embedding_count    = 1000
          leaf_nodes_to_search_percent = 10
        }
      }
    }
  }

  index_update_method = "STREAM_UPDATE"

  labels = merge(var.labels, {
    component = "vector-search"
  })
}

# ------------------------------
# Index Endpoint
# ------------------------------

resource "google_vertex_ai_index_endpoint" "medical_rag" {
  project      = var.project_id
  region       = var.region
  display_name = local.endpoint_name
  description  = "Endpoint for medical RAG vector search (${var.environment})"

  network = var.environment == "prod" ? "projects/${var.project_id}/global/networks/default" : null

  labels = merge(var.labels, {
    component = "vector-search-endpoint"
  })
}

# ------------------------------
# NOTE: Index deployment must be done separately
# after the index has been populated with data.
# Use gcloud or the Python SDK to deploy:
#
# gcloud ai index-endpoints deploy-index ENDPOINT_ID \
#   --deployed-index-id=DEPLOYED_INDEX_ID \
#   --index=INDEX_ID \
#   --display-name="Medical RAG Index" \
#   --machine-type=n1-standard-4 \
#   --min-replica-count=1 \
#   --max-replica-count=2
# ------------------------------

# ------------------------------
# Outputs
# ------------------------------

output "index_id" {
  description = "ID of the Vector Search index"
  value       = google_vertex_ai_index.medical_rag.id
}

output "index_name" {
  description = "Name of the Vector Search index"
  value       = google_vertex_ai_index.medical_rag.name
}

output "endpoint_id" {
  description = "ID of the Index Endpoint"
  value       = google_vertex_ai_index_endpoint.medical_rag.id
}

output "endpoint_name" {
  description = "Name of the Index Endpoint"
  value       = google_vertex_ai_index_endpoint.medical_rag.name
}

output "endpoint_public_domain" {
  description = "Public domain of the endpoint"
  value       = google_vertex_ai_index_endpoint.medical_rag.public_endpoint_domain_name
}
