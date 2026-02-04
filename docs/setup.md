# Setup Guide

This guide walks through setting up the Multimodal Medical RAG system from scratch.

## Prerequisites

### Required Software

- **Python 3.10+**: [Download](https://www.python.org/downloads/)
- **Google Cloud SDK**: [Install Guide](https://cloud.google.com/sdk/docs/install)
- **Terraform 1.5+**: [Install Guide](https://developer.hashicorp.com/terraform/install)
- **Git**: [Download](https://git-scm.com/downloads)

### GCP Requirements

- GCP project with billing enabled
- Owner or Editor role on the project
- APIs enabled (see below)

## Step 1: GCP Project Setup

### Enable Required APIs

```bash
# Set your project
gcloud config set project YOUR_PROJECT_ID

# Enable all required APIs
gcloud services enable \
    storage-api.googleapis.com \
    aiplatform.googleapis.com \
    bigquery.googleapis.com \
    cloudfunctions.googleapis.com \
    run.googleapis.com \
    cloudbuild.googleapis.com \
    compute.googleapis.com
```

### Create Service Account (Optional)

For local development without user credentials:

```bash
# Create service account
gcloud iam service-accounts create medical-rag-dev \
    --display-name="Medical RAG Development"

# Grant permissions
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:medical-rag-dev@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/editor"

# Create and download key
gcloud iam service-accounts keys create ./service-account-key.json \
    --iam-account=medical-rag-dev@YOUR_PROJECT_ID.iam.gserviceaccount.com
```

## Step 2: Local Environment Setup

### Clone and Setup Virtual Environment

```bash
# Navigate to project
cd multimodal-medical-rag-gcp

# Create virtual environment
python -m venv .venv

# Activate (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Activate (Linux/macOS)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Configure Environment Variables

```bash
# Copy example file
cp .env.example environments/dev/.env

# Edit with your values
# Required:
#   - GCP_PROJECT_ID
#   - GCP_REGION
```

### Authenticate with GCP

```bash
# Option 1: User credentials (recommended for development)
gcloud auth application-default login

# Option 2: Service account
export GOOGLE_APPLICATION_CREDENTIALS=./service-account-key.json
```

## Step 3: Deploy Infrastructure

### Initialize Terraform

```bash
cd terraform/environments/dev

# Initialize providers
terraform init
```

### Review and Apply

```bash
# Preview changes
terraform plan -var-file="terraform.tfvars"

# Apply (type 'yes' to confirm)
terraform apply -var-file="terraform.tfvars"
```

### Save Output Values

```bash
# Get the created resource IDs
terraform output

# Update your .env with:
#   - VERTEX_VECTOR_SEARCH_INDEX_ENDPOINT (from endpoint_id)
```

## Step 4: Verify Setup

### Test GCP Connection

```bash
# Run integration tests
pytest tests/integration/test_gcp_connection.py -v
```

### Generate Sample Data

```bash
# Generate synthetic reports
python -m src.data_ingestion.generate_synthetic

# Check data/raw/synthetic/ for output
```

### Test Embedding Generation

```bash
# Requires deployed infrastructure
python -m src.embeddings.text_embeddings
```

## Step 5: Load Initial Data

### Upload Sample Data to GCS

```bash
# Using gsutil
gsutil -m cp -r data/raw/synthetic/* gs://YOUR_PROJECT_ID-raw-data-dev/synthetic/
```

### Generate and Index Embeddings

```bash
# Generate embeddings
python -m src.embeddings.text_embeddings

# Upload to vector search (after index deployment)
# See Terraform outputs for index endpoint
```

## Troubleshooting

### Common Issues

**Authentication errors**:
```bash
# Re-authenticate
gcloud auth application-default login
gcloud auth application-default set-quota-project YOUR_PROJECT_ID
```

**API not enabled**:
```bash
# Check enabled APIs
gcloud services list --enabled

# Enable missing API
gcloud services enable SERVICE_NAME.googleapis.com
```

**Terraform state issues**:
```bash
# Refresh state
terraform refresh

# If corrupted, import resources manually
terraform import google_storage_bucket.raw BUCKET_NAME
```

### Getting Help

- Check [GCP Documentation](https://cloud.google.com/docs)
- Review [Vertex AI Guides](https://cloud.google.com/vertex-ai/docs)
- Open an issue in the repository

## Next Steps

1. Deploy to test environment
2. Configure CI/CD pipelines
3. Set up monitoring and alerts
4. Load production data (with proper access controls)
