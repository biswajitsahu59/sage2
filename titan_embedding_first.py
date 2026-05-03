import boto3
import numpy as np
import json
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

# ================================
# Titan Bedrock Setup
# ================================
bedrock_client = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")

MODEL_ID = "amazon.titan-embed-text-v1"

# ================================
# Generate embedding for one text with retry
# ================================
def generate_single_embedding(text, retries=3, delay=2):
    """
    Generate embedding for a single text with retry logic.
    """
    for attempt in range(1, retries + 1):
        try:
            payload = {
                "inputText": text  # Titan expects ONE string
            }

            response = bedrock_client.invoke_model(
                modelId=MODEL_ID,
                body=json.dumps(payload)
            )

            body = json.loads(response["body"].read())
            return body["embedding"]

        except Exception as e:
            print(f"Attempt {attempt} failed for text '{text[:30]}...': {e}")
            if attempt < retries:
                time.sleep(delay * attempt)  # Exponential backoff
            else:
                return None  # If it fails after max retries, return None

# ================================
# Parallel Embedding Function
# ================================
def create_titan_embeddings(data, max_workers=10):
   
    if isinstance(data, pd.Series):
        data = data.tolist()
    elif isinstance(data, pd.DataFrame):
        raise ValueError("Provide Series or list, not DataFrame.")

    total_records = len(data)
    print(f"Total records: {total_records}")
    print(f"Processing with {max_workers} workers...")

    results = [None] * total_records

    def process_record(idx, text):
        results[idx] = generate_single_embedding(text)

    # Run parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_record, i, data[i]) for i in range(total_records)]
        for future in tqdm(as_completed(futures), total=total_records, desc="Embedding Progress"):
            future.result()

    # Count missing embeddings
    missing_count = sum(1 for e in results if e is None)
    if missing_count > 0:
        print(f"Warning: {missing_count} records failed to generate embeddings.")
    else:
        print("All embeddings generated successfully.")

    # Convert to NumPy array for FAISS if all embeddings are present
    valid_embeddings = [e for e in results if e is not None]
    if valid_embeddings:
        return np.vstack(valid_embeddings).astype('float32')
    return results