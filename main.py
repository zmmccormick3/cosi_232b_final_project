import json
from datasets import load_dataset
from transformers import BertTokenizer
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Step 1: Load the Dataset
ds = load_dataset("brainchalov/pubmed_arxiv_abstracts_data")
train_data = ds["train"]

# Step 2: Tokenize and Chunk the Text into 300-token chunks
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def tokenize_and_chunk(example):
    # Tokenize the abstract
    tokens = tokenizer(example["abstr"], truncation=True, padding="max_length", max_length=300, return_tensors="pt")
    return {'tokens': tokens['input_ids'].squeeze(0).tolist()}


# Apply the tokenization and chunking
train_data = train_data.map(tokenize_and_chunk, batched=False)

train_data.save_to_disk('processed_train_data_300')


# Extract the tokenized chunks for indexing
passages = [(i, train_data[i]['abstr']) for i in range(len(train_data))]

# Step 3: Build the FAISS Index

# Initialize the sentence transformer model
embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Convert the abstract chunks into embeddings
passage_texts = [p[1] for p in passages]
passage_embeddings = embedder.encode(passage_texts, convert_to_numpy=True)

# Build a FAISS index
dimension = passage_embeddings.shape[1]  # Embedding size, e.g., 384
index = faiss.IndexFlatIP(dimension)  # Using inner product similarity
index.add(passage_embeddings)

# Store metadata (passage IDs, texts) to retrieve later
# metadata = [{"id": i, "text": passage_texts[i]} for i in range(len(passage_texts))]

metadata = []
for i in range(len(passage_texts)):
    metadata.append({
        "id": i,
        "text": passage_texts[i],
        "title": train_data[i]["title"],
        "journal": train_data[i]["journal"],
        "field": train_data[i]["field"]
    })


# Save the index and metadata (optional)
faiss.write_index(index, 'faiss_index.index')
with open('metadata.json', 'w') as f:
    json.dump(metadata, f)


# Step 4: Define the Retriever Function to Retrieve Passages for a Query

def retrieve_passages(query, top_k=5):
    # Encode the query
    query_embedding = embedder.encode([query], convert_to_numpy=True)

    # Search the FAISS index for the top-k most similar passages
    distances, indices = index.search(query_embedding, top_k)

    # Retrieve the actual text of the top-k passages
    results = []
    for i in indices[0]:
        # passage_text = metadata[i]['text']
        # results.append(passage_text)

        passage_info = metadata[i]
        # You could include more fields like title if you want
        results.append(f"Title: {passage_info['title']}\nAbstract: {passage_info['text']}\n")

    return results


# Step 5: Design a Prompt for LLM with Retrieved Context

def create_prompt(query):
    # Retrieve top 5 relevant passages
    retrieved_passages = retrieve_passages(query, top_k=5)

    # Combine the query with the retrieved passages to form a prompt
    prompt = f"Query: {query}\n\n"
    prompt += "Context:\n"
    prompt += "\n".join(retrieved_passages)

    return prompt

embedder.save('sentence_transformer_model_300')

# Example: Use the function to create a prompt
# query = "What are the applications of predictive coding in auditory processing?"

# query = input("Enter your query: ")
# prompt = create_prompt(query)
# print("Generated Prompt for LLM:\n")
# print(prompt)
