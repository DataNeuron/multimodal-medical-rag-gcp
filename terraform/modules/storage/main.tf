# ============================================
# Cloud Storage Module
# ============================================
# Creates GCS buckets for data storage with proper configuration
#
# Buckets created:
#   - Raw data bucket: Original medical images and documents
#   - Processed bucket: Preprocessed and cleaned data
#   - Embeddings bucket: Generated embedding vectors

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
  description = "GCP region for bucket location"
  type        = string
  default     = "us-central1"
}

variable "environment" {
  description = "Environment name (dev/test/prod)"
  type        = string
}

variable "storage_class" {
  description = "Storage class for buckets"
  type        = string
  default     = "STANDARD"
}

variable "lifecycle_days" {
  description = "Days before objects are deleted (0 = disabled)"
  type        = number
  default     = 0
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
  bucket_prefix = "${var.project_id}"

  buckets = {
    raw = {
      name        = "${local.bucket_prefix}-raw-data-${var.environment}"
      description = "Raw medical data (images, documents)"
    }
    processed = {
      name        = "${local.bucket_prefix}-processed-${var.environment}"
      description = "Processed and cleaned data"
    }
    embeddings = {
      name        = "${local.bucket_prefix}-embeddings-${var.environment}"
      description = "Generated embedding vectors"
    }
  }
}

# ------------------------------
# Raw Data Bucket
# ------------------------------

resource "google_storage_bucket" "raw" {
  name          = local.buckets.raw.name
  project       = var.project_id
  location      = var.region
  storage_class = var.storage_class

  uniform_bucket_level_access = true

  versioning {
    enabled = var.environment == "prod" ? true : false
  }

  dynamic "lifecycle_rule" {
    for_each = var.lifecycle_days > 0 ? [1] : []
    content {
      condition {
        age = var.lifecycle_days
      }
      action {
        type = "Delete"
      }
    }
  }

  labels = merge(var.labels, {
    bucket-type = "raw-data"
  })
}

# ------------------------------
# Processed Data Bucket
# ------------------------------

resource "google_storage_bucket" "processed" {
  name          = local.buckets.processed.name
  project       = var.project_id
  location      = var.region
  storage_class = var.storage_class

  uniform_bucket_level_access = true

  versioning {
    enabled = false
  }

  dynamic "lifecycle_rule" {
    for_each = var.lifecycle_days > 0 ? [1] : []
    content {
      condition {
        age = var.lifecycle_days
      }
      action {
        type = "Delete"
      }
    }
  }

  labels = merge(var.labels, {
    bucket-type = "processed"
  })
}

# ------------------------------
# Embeddings Bucket
# ------------------------------

resource "google_storage_bucket" "embeddings" {
  name          = local.buckets.embeddings.name
  project       = var.project_id
  location      = var.region
  storage_class = var.storage_class

  uniform_bucket_level_access = true

  versioning {
    enabled = false
  }

  labels = merge(var.labels, {
    bucket-type = "embeddings"
  })
}

# ------------------------------
# Outputs
# ------------------------------

output "raw_bucket_name" {
  description = "Name of the raw data bucket"
  value       = google_storage_bucket.raw.name
}

output "raw_bucket_url" {
  description = "URL of the raw data bucket"
  value       = google_storage_bucket.raw.url
}

output "processed_bucket_name" {
  description = "Name of the processed data bucket"
  value       = google_storage_bucket.processed.name
}

output "embeddings_bucket_name" {
  description = "Name of the embeddings bucket"
  value       = google_storage_bucket.embeddings.name
}

output "bucket_names" {
  description = "Map of all bucket names"
  value = {
    raw        = google_storage_bucket.raw.name
    processed  = google_storage_bucket.processed.name
    embeddings = google_storage_bucket.embeddings.name
  }
}
