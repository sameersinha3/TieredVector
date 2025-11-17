# System Design

## Cloud Storage (for now)
Tier 3 (cold storage) will in the future be massive. Thus it makes only makes sense to use vector search using a VectorDB. Vertex AI is not free, and while AWS is free, it does have usage limits for the free tier. Instead, I have decided to simulate cloud storage by spinning up a VM on CloudLab, starting ChromaDB and connecting to it. Here are the steps

First start a CloudLab experiment and ssh into the VM Then get the IP of the VM. Then paste the output into your env file with VM_IP as the key

```bash
curl ifconfig.me
```

```
VM_IP=XXX.XX.XXX.X
```

Finally, run the following to install ChromaDB and run it (the specified path should use the saved ChromaDB store that was the result of simulate_temperature.py. I WILL CONFIRM THIS TOMORROW TO SEE IF IT PERSISTS)
```bash
sudo apt update
sudo apt install python3-pip
pip install chromadb
export PATH="$PATH:/users/sameers5/.local/bin"
echo 'export PATH="$PATH:/users/sameers5/.local/bin"' >> ~/.bashrc
source ~/.bashrc
chroma run --host 0.0.0.0 --port 8000 --path /users/sameers5/chroma_store
```
Now Chroma is running.
Then, you can run a query in sandbox.py by replacing the query string. 