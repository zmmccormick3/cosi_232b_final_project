import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os

# Load saved FAISS index and metadata
index = faiss.read_index('faiss_index.index')
with open('metadata.json', 'r') as f:
    metadata = json.load(f)
embedder = SentenceTransformer('sentence_transformer_model_300')

def retrieve_passages(query, top_k=5):
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    results = []
    for i in indices[0]:
        passage_info = metadata[i]
        results.append(f"Title: {passage_info['title']}\nAbstract: {passage_info['text']}\n")
    return results

queries = [
    "Applications of predictive coding in auditory processing",
    "Synaptic plasticity and memory formation",
    "Advances in Alzheimer's disease treatment",
    "Mechanism of vesicle fusion in neurons"
]

# Automatically find a new filename if needed
base_filename = "query_results.txt"
filename = base_filename
counter = 1
while os.path.exists(filename):
    filename = f"query_results_{counter}.txt"
    counter += 1

# Now use 'filename' instead of hardcoding 'query_results.txt'
with open(filename, 'w') as out_file:
    for query in queries:
        out_file.write(f"Query: {query}\n")
        retrieved_passages = retrieve_passages(query, top_k=5)
        out_file.write("Context:\n")
        for passage in retrieved_passages:
            out_file.write(passage + "\n")
        out_file.write("\n" + "="*80 + "\n\n")
