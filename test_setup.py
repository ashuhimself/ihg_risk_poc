# test_setup.py
from google.cloud import bigquery, storage, aiplatform

# Test BigQuery
print("Testing BigQuery...")
bq_client = bigquery.Client(project="ihg-mlops")
query = "SELECT COUNT(*) as count FROM `ihg-mlops.ihg_training_data.booking` LIMIT 1"
result = bq_client.query(query).result()
for row in result:
    print(f"✓ BigQuery works! Row count: {row.count}")

# Test GCS
print("\nTesting Cloud Storage...")
storage_client = storage.Client(project="ihg-mlops")
bucket = storage_client.bucket("ihg-mlops")
print(f"✓ GCS works! Bucket: {bucket.name}")

# Test Vertex AI
print("\nTesting Vertex AI...")
aiplatform.init(project="ihg-mlops", location="us-central1")
print("✓ Vertex AI initialized!")

print("\n✅ All services are accessible!")