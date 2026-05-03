import boto3
import numpy as np
import pandas as pd
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from botocore.exceptions import ClientError

# ---------- CONFIGURATION ----------
REGION = "us-east-1"  # Update region as per your Bedrock setup
MAX_RETRIES = 10        # Retry attempts for throttling/network errors
RETRY_BACKOFF = 2      # Backoff multiplier (exponential)
BATCH_SIZE = 32        # Number of texts processed in each batch
MAX_WORKERS = 10       # Parallel threads


# ---------- EMBEDDING FUNCTION ----------
def generate_single_embedding(text, bedrock_client, model_id):
    """
    Generate embedding for a single text using Titan.
    Includes retry logic for throttling/network errors.
    """
    payload = {"inputText": text}

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = bedrock_client.invoke_model(
                modelId=model_id,
                body=json.dumps(payload),
                contentType="application/json",
                accept="application/json"
            )

            response_body = json.loads(response["body"].read())
            return response_body["embedding"]

        except ClientError as e:
            error_code = e.response["Error"]["Code"]

            # Handle throttling gracefully
            if error_code in ["ThrottlingException", "TooManyRequestsException"]:
                sleep_time = RETRY_BACKOFF ** attempt
                print(f"[Throttle] Retrying in {sleep_time}s (attempt {attempt})...")
                time.sleep(sleep_time)
            else:
                print(f"[Error] Non-throttling error: {str(e)}")
                return None

        except Exception as e:
            print(f"[Error] {str(e)} (attempt {attempt})")
            time.sleep(RETRY_BACKOFF ** attempt)

    print("[Error] Max retries exceeded for:", text[:50])
    return None


# ---------- PARALLEL EMBEDDING CREATION ----------


import boto3
import numpy as np
import pandas as pd
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Constants
REGION = "us-east-1"
BATCH_SIZE = 32


def create_titan_embeddings(text_list, model_id, batch_size=BATCH_SIZE, max_workers=MAX_WORKERS, save_path="raw_embeddings.parquet"):
    total_records = len(text_list)
    print(f"Total records: {total_records}")
    print(f"Processing with {max_workers} workers, batch size {batch_size}...")
    print(f"Using model: {model_id}")

    # Set Titan embedding size
    EMBEDDING_SIZE = 1024  # Default for Titan model
    print(f"[Info] Using embedding size: {EMBEDDING_SIZE}")

    # Initialize Bedrock client
    bedrock_client = boto3.client("bedrock-runtime", region_name=REGION)

    # Prepare storage for embeddings
    embeddings = [None] * total_records

    # Function to generate a single embedding
    def process_record(idx, text):
        try:
            emb = generate_single_embedding(text, bedrock_client, model_id)
            embeddings[idx] = emb
        except Exception as e:
            print(f"[Error] Record {idx} failed: {e}")
            embeddings[idx] = None

    # Run multithreaded processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_record, i, text_list[i]) for i in range(total_records)]
        for _ in tqdm(as_completed(futures), total=total_records, desc="Embedding Progress"):
            pass

    # Count missing embeddings
    missing = sum(1 for e in embeddings if e is None)
    if missing > 0:
        print(f"[Warning] {missing} embeddings failed and returned None.")
    else:
        print("[Success] All embeddings generated successfully!")

    # Step 1: Save raw embeddings before cleaning
    print(f"Saving raw embeddings to {save_path}...")
    raw_df = pd.DataFrame({
        "text": text_list,
        "embedding_raw": [json.dumps(e) if e is not None else None for e in embeddings]
    })
    raw_df.to_parquet(save_path, index=False)
    print(f"[Saved] Raw embeddings saved to {save_path}")

    # Step 2: Clean embeddings
    clean_embeddings = []
    for i, e in enumerate(embeddings):
        if e is None:
            # Completely missing embedding
            print(f"[Warning] Record {i}: None found, replacing with zeros.")
            clean_embeddings.append([0.0] * EMBEDDING_SIZE)

        elif isinstance(e, list):
            # Handle nested list like [[...]]
            if len(e) == 1 and isinstance(e[0], list):
                e = e[0]

            if len(e) != EMBEDDING_SIZE:
                print(f"[Warning] Record {i}: Invalid size {len(e)}, expected {EMBEDDING_SIZE}. Replacing with zeros.")
                clean_embeddings.append([0.0] * EMBEDDING_SIZE)
            else:
                clean_embeddings.append(e)

        else:
            # If it's not a list at all (maybe a string or dict)
            print(f"[Warning] Record {i}: Unexpected type {type(e)}, replacing with zeros.")
            clean_embeddings.append([0.0] * EMBEDDING_SIZE)

    # Convert to numpy array
    final_embeddings = np.array(clean_embeddings, dtype="float32")
    print("Final embeddings shape:", final_embeddings.shape)

    return final_embeddings



