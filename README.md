# System Design

## Cloud Storage (for now)
Tier 3 (cold storage) will in the future be massive. Thus it makes only makes sense to use vector search using a VectorDB. Vertex AI is not free, and while AWS is free, it does have usage limits for the free tier. Instead, I have decided to simulate cloud storage by spinning up a VM on CloudLab, starting ChromaDB and connecting to it. The steps to this are to start Chroma

```bash
sudo apt update
sudo apt install python3-pip
pip install chromadb
export PATH="$PATH:/users/sameers5/.local/bin"
echo 'export PATH="$PATH:/users/sameers5/.local/bin"' >> ~/.bashrc
source ~/.bashrc
chroma run --host 0.0.0.0 --port 8000 --path ./chroma_data
```
Now Chroma is running. We need to retrieve the VM's IP as follows
```bash
curl ifconfig.me
```

Then paste the output into your env file with VM_IP as the key

Now you should be able to run simulate_temperature.py which places the vectors across the three tiers (local redis and local LMDB). If you haven't, you will need to run load_queries.py and load_dataset.py once to get the *_embeddings.npy files (this can take a while). 

Then, you can run a query in sandbox.py by replacing the query string.