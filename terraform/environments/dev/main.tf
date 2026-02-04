# ============================================
# Development Environment Infrastructure
# ============================================
# Deploys all resources for the dev environment
#
# Usage:
#   terraform init
#   terraform plan -var-file="terraform.tfvars"
#   terraform apply -var-file="terraform.tfvars"

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }

  # Backend configuration - uncomment for remote state
  # backend "gcs" {
  #   bucket = "multimodal-medical-rag-tfstate"
  #   prefix = "terraform/dev"
  # }
}

# ------------------------------
# Provider Configuration
# ------------------------------

provider "google" {
  project = var.project_id
  region  = var.region
}

# ------------------------------
# Variables
# ------------------------------

variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "dev"
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "GCP zone"
  type        = string
  default     = "us-central1-a"
}

variable "storage_class" {
  description = "GCS storage class"
  type        = string
  default     = "STANDARD"
}

variable "storage_lifecycle_days" {
  description = "Days before object deletion"
  type        = number
  default     = 30
}

variable "vertex_ai_machine_type" {
  description = "Machine type for Vertex AI"
  type        = string
  default     = "n1-standard-4"
}

variable "vector_search_shard_size" {
  description = "Shard size for vector search"
  type        = string
  default     = "SHARD_SIZE_SMALL"
}

variable "bq_delete_contents_on_destroy" {
  description = "Allow BQ dataset deletion"
  type        = bool
  default     = true
}

variable "enable_private_google_access" {
  description = "Enable private Google access"
  type        = bool
  default     = false
}

variable "enable_budget_alerts" {
  description = "Enable budget alerts"
  type        = bool
  default     = true
}

variable "budget_amount" {
  description = "Monthly budget in USD"
  type        = number
  default     = 100
}

variable "labels" {
  description = "Labels for all resources"
  type        = map(string)
  default     = {}
}

# ------------------------------
# Modules
# ------------------------------

module "storage" {
  source = "../../modules/storage"

  project_id     = var.project_id
  region         = var.region
  environment    = var.environment
  storage_class  = var.storage_class
  lifecycle_days = var.storage_lifecycle_days
  labels         = var.labels
}

module "vertex_ai" {
  source = "../../modules/vertex-ai"

  project_id          = var.project_id
  region              = var.region
  environment         = var.environment
  embedding_dimension = 768
  shard_size          = var.vector_search_shard_size
  machine_type        = var.vertex_ai_machine_type
  labels              = var.labels
}

module "bigquery" {
  source = "../../modules/bigquery"

  project_id                 = var.project_id
  region                     = var.region
  environment                = var.environment
  delete_contents_on_destroy = var.bq_delete_contents_on_destroy
  labels                     = var.labels
}

# ------------------------------
# Outputs
# ------------------------------

output "storage_buckets" {
  description = "Created storage bucket names"
  value       = module.storage.bucket_names
}

output "vector_search_index_id" {
  description = "Vector Search index ID"
  value       = module.vertex_ai.index_id
}

output "vector_search_endpoint_id" {
  description = "Vector Search endpoint ID"
  value       = module.vertex_ai.endpoint_id
}

output "bigquery_dataset_id" {
  description = "BigQuery dataset ID"
  value       = module.bigquery.dataset_id
}

output "environment_info" {
  description = "Environment configuration summary"
  value = {
    project_id  = var.project_id
    environment = var.environment
    region      = var.region
  }
}
