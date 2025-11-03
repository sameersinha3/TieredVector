import numpy as np
import os

from dotenv import load_dotenv
from storage_manager import StorageManager

load_dotenv()
manager = StorageManager(
    gcs_project="VectorTier",
    gcs_bucket=os.getenv("BUCKET_NAME"),
    gcs_blob_name=os.getenv("BLOB"),
    sa_key_path=os.getenv("SA_KEY")
)

manager.initialize()

embedding = np.random.rand(768).astype(np.float32)
manager.store_document(doc_id=42, embedding=embedding, temperature=0.5)

query = np.random.rand(768).astype(np.float32)
results = manager.retrieve_document(query, k=5, threshold=0.3) # Low threshold since our query is just a randint
print(results)

manager.close()