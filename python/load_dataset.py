from datasets import load_dataset
import numpy as np

'''
Code to load the dataset and save it in a .npy file (not necessary)
'''

# Load the dataset and save it in a .npy file for future access
dataset = load_dataset("Cohere/wikipedia-22-12-simple-embeddings", split="train")
embeddings = np.array(dataset["emb"]) 
np.save("wiki_embeddings.npy", embeddings)