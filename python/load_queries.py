import cohere
from dotenv import load_dotenv
import itertools
import numpy as np
import os

from datasets import load_dataset

load_dotenv()
API_KEY = os.getenv("COHERE_API_KEY")
co = cohere.Client(API_KEY)


query_dataset = load_dataset("natural_questions", split="train", streaming=True)
queries = [entry["question"]["text"] for entry in itertools.islice(query_dataset, 50000)]
print(queries[0])

print(f"Number of examples in the dataset: {len(queries)}")

response = co.embed(
    model="multilingual-22-12",
    texts=queries[:50000]
)

query_embeddings = np.array(response.embeddings)


np.save("query_embeddings.npy", query_embeddings)

print("Saved to query_embeddings.py")

