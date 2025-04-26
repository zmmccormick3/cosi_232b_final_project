import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import json

# Load the FAISS index and metadata
index = faiss.read_index('faiss_index.index')

with open('metadata.json', 'r') as f:
    metadata = json.load(f)

# Initialize the SentenceTransformer model
embedder = SentenceTransformer('sentence_transformers/all-MiniLM-L6-v2')


# Function to retrieve passages based on query
def retrieve_passages(query, top_k=5):
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for i in indices[0]:
        passage_info = metadata[i]
        results.append(f"Title: {passage_info['title']}\nAbstract: {passage_info['text']}\n")
    return results


# Function to create a prompt for LLM
def create_prompt(query):
    retrieved_passages = retrieve_passages(query, top_k=5)
    prompt = f"Query: {query}\n\n"
    prompt += "Context:\n"
    prompt += "\n".join(retrieved_passages)
    return prompt


# Streamlit Interface
st.title("Research Paper Query System")

# Input query from user
query = st.text_input("Enter your query:")

if query:
    prompt = create_prompt(query)
    st.subheader("Generated Prompt for LLM")
    st.write(prompt)



# streamlit run streamlit_app.py