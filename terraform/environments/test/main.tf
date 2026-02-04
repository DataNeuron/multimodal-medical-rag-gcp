# ============================================
# Test Environment Infrastructure
# ============================================
# Identical structure to dev - see dev/main.tf for documentation

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

variable "project_id" { type = string }
variable "environment" { type = string; default = "test" }
variable "region" { type = string; default = "us-central1" }
variable "zone" { type = string; default = "us-central1-a" }
variable "storage_class" { type = string; default = "STANDARD" }
variable "storage_lifecycle_days" { type = number; default = 14 }
variable "vertex_ai_machine_type" { type = string; default = "n1-standard-8" }
variable "vector_search_shard_size" { type = string; default = "SHARD_SIZE_SMALL" }
variable "bq_delete_contents_on_destroy" { type = bool; default = true }
variable "enable_private_google_access" { type = bool; default = true }
variable "enable_budget_alerts" { type = bool; default = true }
variable "budget_amount" { type = number; default = 250 }
variable "labels" { type = map(string); default = {} }

module "storage" {
  source         = "../../modules/storage"
  project_id     = var.project_id
  region         = var.region
  environment    = var.environment
  storage_class  = var.storage_class
  lifecycle_days = var.storage_lifecycle_days
  labels         = var.labels
}

module "vertex_ai" {
  source              = "../../modules/vertex-ai"
  project_id          = var.project_id
  region              = var.region
  environment         = var.environment
  embedding_dimension = 768
  shard_size          = var.vector_search_shard_size
  machine_type        = var.vertex_ai_machine_type
  labels              = var.labels
}

module "bigquery" {
  source                     = "../../modules/bigquery"
  project_id                 = var.project_id
  region                     = var.region
  environment                = var.environment
  delete_contents_on_destroy = var.bq_delete_contents_on_destroy
  labels                     = var.labels
}

output "storage_buckets" { value = module.storage.bucket_names }
output "vector_search_index_id" { value = module.vertex_ai.index_id }
output "vector_search_endpoint_id" { value = module.vertex_ai.endpoint_id }
output "bigquery_dataset_id" { value = module.bigquery.dataset_id }
