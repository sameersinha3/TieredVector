import cohere
import numpy as np
import os

from dotenv import load_dotenv
from storage_manager import StorageManager

load_dotenv()

manager = StorageManager(
    chroma_host=os.getenv("VM_IP"),
    chroma_port=8000,
    chroma_collection="cold_vectors" 
)

manager.initialize()

query = "When was youtube founded"
co = cohere.Client(os.getenv("COHERE_API_KEY"))
query_embedding = np.array(co.embed(
    model="multilingual-22-12",
    texts=[query]
).embeddings[0])

results = manager.retrieve_document(query_embedding, k=5, threshold=0.8)
print(results)

manager.close()