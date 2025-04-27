import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import os
import torch    # for LLaMa

# Load saved FAISS index and metadata
index = faiss.read_index('faiss_index.index')
with open('metadata.json', 'r') as f:
    metadata = json.load(f)
embedder = SentenceTransformer('sentence_transformer_model_300')

# Load generation model
## if with FLAN
# generator = pipeline("text2text-generation", model="google/flan-t5-base")

## if with LLaMa
# Load LLaMA model (llama3.1)
generator = pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B")  # Correct LLaMa model name



def retrieve_passages(query, top_k=5):
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    results = []
    for i in indices[0]:
        passage_info = metadata[i]
        results.append(f"Title: {passage_info['title']}\nAbstract: {passage_info['text']}\n")
    print(results)
    return results

def generate_answer(query, contexts):
    # Combine contexts into one string
    combined_context = " ".join(contexts)
    # Create a prompt
    prompt = f"Answer the following question based on the context provided.\n\nContext: {combined_context}\n\nQuestion: {query}\n\nAnswer:"
    # Generate
    ## if with FLAN
    # result = generator(prompt, max_length=500, do_sample=True, temperature=0.7, top_p=0.9)
    
    ## if with LLaMa
    # result = generator(prompt, max_length=500, num_return_sequences=1, temperature=0.7, top_p=0.9)
    # Generate text using the GPT-Neo model
    result = generator(prompt, do_sample=True, min_length=50)  # Ensure do_sample is set to True for more diverse generation
    print(result)
    return result[0]['generated_text']

queries = [
    # Systems Neuroscience
    "How do auditory pathways encode the frequency of sound?",
    "What role does the thalamus play in sensory information processing?",
    "Compare the dorsal and ventral visual streams.",
    "How do different types of retinal ganglion cells contribute to vision?",
    "Is the somatosensory cortex organized somatotopically?",
    "Explain the function of the superior colliculus in motor behavior.",
    "What is the significance of lateral inhibition in sensory systems?",
    "How does the olfactory system bypass the thalamus?",
    "Can the vestibular system influence visual perception?",
    "Describe how the hippocampus processes spatial information.",
    # Electrophysiology
    "What causes the refractory period of an action potential?",
    "How is patch-clamp recording used to study ion channel behavior?",
    "Are sodium and potassium channels voltage-gated?",
    "Explain the concept of reversal potential.",
    "Describe the difference between excitatory and inhibitory postsynaptic potentials (EPSPs and IPSPs).",
    "How do calcium spikes differ from sodium spikes?",
    "Can neurons fire without action potentials?",
    "What is the role of afterhyperpolarization in neuronal firing?",
    "How do local field potentials reflect population-level neural activity?",
    "What electrophysiological markers are used to detect synaptic plasticity?",
    # Computational Neuroscience
    "What is sparse coding in the context of sensory processing?",
    "How do rate coding and temporal coding differ?",
    "Can a single neuron act as a logic gate?",
    "Explain the basics of predictive coding models in the brain.",
    "How are attractor networks used to model memory?",
    "What does the Hodgkin-Huxley model simulate?",
    "Are recurrent neural networks biologically plausible?",
    "How can convolutional neural networks be inspired by the visual cortex?",
    "Describe the role of Bayesian inference in perceptual decision making.",
    "What is the importance of noise in neural computation?",
    # Cognitive Neuroscience
    "What brain areas are critical for working memory?",
    "How does selective attention modulate sensory processing?",
    "Is the prefrontal cortex involved in planning?",
    "Explain the role of the hippocampus in episodic memory.",
    "What evidence supports the distributed nature of semantic memory?",
    "Can language production occur without Broca's area?",
    "Describe the neural basis of cognitive control.",
    "How does the brain resolve sensory conflict (e.g., McGurk effect)?",
    "Are there critical periods for language learning?",
    "What is the default mode network and when is it active?",
    # Clinical Neuroscience
    "What are the hallmark pathological features of Alzheimer's disease?",
    "How is deep brain stimulation used to treat Parkinson's disease?",
    "Can demyelination occur without motor symptoms?",
    "What treatments exist for drug-resistant epilepsy?",
    "Describe the neurobiological basis of depression.",
    "Is neurogenesis possible in the adult human brain?",
    "How do stroke-induced lesions affect language function?",
    "What mechanisms underlie multiple sclerosis progression?",
    "Can traumatic brain injury lead to long-term cognitive impairment?",
    "What are current challenges in treating glioblastoma?"
]


# Automatically find a new filename if needed
base_filename = "query_results.txt"
filename = base_filename
counter = 1
while os.path.exists(filename):
    filename = f"query_results_{counter}.txt"
    counter += 1

with open(filename, 'w') as out_file:
    for query in queries:
        out_file.write(f"Write a detailed multi-sentence answer to the query below, based on the context.\nQuery: {query}\n")

        retrieved_passages = retrieve_passages(query, top_k=5)
        out_file.write("Context:\n")
        for passage in retrieved_passages:
            out_file.write(passage + "\n")
        
        # NEW: Generate answer
        generated_answer = generate_answer(query, retrieved_passages)
        out_file.write("\nGenerated Answer:\n")
        out_file.write(generated_answer + "\n")
        
        out_file.write("\n" + "="*80 + "\n\n")
