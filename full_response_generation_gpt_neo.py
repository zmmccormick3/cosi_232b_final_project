import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import os
import torch # Keep torch import

# --- Configuration ---
MODEL_NAME = "EleutherAI/gpt-neo-1.3B"
SENTENCE_TRANSFORMER_MODEL = 'sentence_transformer_model_300' # Ensure this path is correct
FAISS_INDEX_PATH = 'faiss_index.index'
METADATA_PATH = 'metadata.json'
OUTPUT_FILENAME_BASE = "query_results_gptneo.txt"
TOP_K_RESULTS = 5
MAX_NEW_TOKENS_GENERATION = 512 # Max tokens for the generated answer
# --- End Configuration ---

# Check for GPU availability
device = 0 if torch.cuda.is_available() else -1 # 0 for first GPU, -1 for CPU
if device == 0:
    print("Using GPU (CUDA)")
else:
    print("Using CPU")

# Load saved FAISS index and metadata
try:
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(METADATA_PATH, 'r') as f:
        metadata = json.load(f)
    embedder = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)
    print("FAISS index, metadata, and sentence transformer loaded successfully.")
except Exception as e:
    print(f"Error loading prerequisites: {e}")
    exit()

# Load generation model and tokenizer
try:
    # Load the tokenizer separately to check context length if needed
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # Load the generation pipeline, explicitly setting the device
    generator = pipeline(
        "text-generation",
        model=MODEL_NAME,
        tokenizer=tokenizer,
        device=device # Use GPU if available, else CPU
    )
    print(f"Generator pipeline for {MODEL_NAME} loaded successfully.")
    # Get model's max input length (often stored in config)
    # Default for GPT-Neo is usually 2048
    model_max_length = tokenizer.model_max_length
    print(f"Model max input sequence length: {model_max_length}")

except Exception as e:
    print(f"Error loading generator model/tokenizer: {e}")
    exit()


def retrieve_passages(query, top_k=TOP_K_RESULTS):
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    results = []
    print(f"\nRetrieving top {top_k} passages for query: '{query}'") # Add some logging
    for i in indices[0]:
        if i < 0 or i >= len(metadata): # Basic check for valid index
             print(f"Warning: Invalid index {i} encountered during retrieval.")
             continue
        passage_info = metadata[i]
        # Ensure text exists, handle potential missing keys gracefully
        title = passage_info.get('title', 'No Title')
        text = passage_info.get('text', 'No Abstract/Text')
        results.append(f"Title: {title}\nAbstract: {text}\n")
        # print(f"  - Retrieved passage (Index {i}): {title[:50]}...") # Optional: log retrieved titles
    return results

def generate_answer(query, contexts):
    # Combine contexts into one string
    combined_context = "\n".join(contexts) # Use newline separation for clarity

    # --- Optional but recommended: Truncate context if needed ---
    # Estimate tokens used by prompt template and query
    prompt_template = f"Answer the following question based on the context provided.\n\nContext: {{context}}\n\nQuestion: {query}\n\nAnswer:"
    prompt_prefix = prompt_template.format(context="") # Template without context
    tokens_prefix = tokenizer(prompt_prefix, return_tensors="pt")["input_ids"].shape[1]

    # Leave some buffer for query tokens and special tokens
    buffer_tokens = 50
    max_context_tokens = model_max_length - tokens_prefix - buffer_tokens

    if max_context_tokens < 100: # Ensure we have a reasonable minimum context
         print(f"Warning: Calculated max context tokens ({max_context_tokens}) is very low. Check model_max_length and prompt structure.")
         max_context_tokens = 100 # Set a minimum floor

    # Tokenize the context
    context_tokens = tokenizer(combined_context, return_tensors="pt", truncation=False)["input_ids"][0] # Don't truncate yet

    # If context is too long, truncate it
    if len(context_tokens) > max_context_tokens:
        print(f"Context is too long ({len(context_tokens)} tokens), truncating to {max_context_tokens} tokens.")
        truncated_token_ids = context_tokens[:max_context_tokens]
        combined_context = tokenizer.decode(truncated_token_ids, skip_special_tokens=True)
    # --- End context truncation ---


    # Create the final prompt
    prompt = prompt_template.format(context=combined_context)

    print(f"Generating answer for query: '{query}'")
    # print(f"Prompt being sent to model (first 200 chars): {prompt[:200]}...") # Debug: Check prompt

    try:
        # Generate using the pipeline
        # **** KEY CHANGE: Added max_new_tokens ****
        result = generator(
            prompt,
            max_new_tokens=MAX_NEW_TOKENS_GENERATION, # Control the length of the generated answer
            do_sample=True,
            temperature=0.7, # Keep your sampling params if they worked before
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id # Important for causal LMs
            # num_return_sequences=1 # Default is 1, so usually not needed
            # min_length=50 # Removed min_length - max_new_tokens is usually sufficient control
        )
        print("Raw generation result:", result) # Debug: print the full result

        # Extract the generated text PART, removing the prompt
        full_generated_text = result[0]['generated_text']
        # The generated text includes the prompt, so find where the prompt ends
        answer_part = full_generated_text[len(prompt):].strip()

        # Sometimes the model might generate extra instruction text or repeat parts of the prompt
        # Add simple post-processing if needed, e.g., remove lines starting with "Context:", "Question:", "Answer:"
        lines = answer_part.split('\n')
        cleaned_lines = [line for line in lines if not line.strip().startswith(('Context:', 'Question:', 'Answer:'))]
        final_answer = "\n".join(cleaned_lines).strip()

        if not final_answer:
             print("Warning: Generated answer was empty after removing prompt/processing.")
             return "[Model generated empty response]" # Return placeholder

        return final_answer

    except Exception as e:
        print(f"Error during text generation for query '{query}': {e}")
        # Consider adding more specific error handling if needed (e.g., for CUDA OOM)
        if "CUDA out of memory" in str(e):
            print("CUDA Out of Memory error detected. Try reducing batch size (if applicable), max_new_tokens, or using a smaller model.")
            # You might want to exit or skip the rest of the queries here
            raise e # Re-raise the exception to stop the script
        return f"[Error during generation: {e}]" # Return error message

# List of queries
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
filename = OUTPUT_FILENAME_BASE
counter = 1
while os.path.exists(filename):
    filename = f"{OUTPUT_FILENAME_BASE.split('.')[0]}_{counter}.{OUTPUT_FILENAME_BASE.split('.')[-1]}"
    counter += 1
print(f"Output will be written to: {filename}")

# Main loop
with open(filename, 'w', encoding='utf-8') as out_file: # Use utf-8 encoding
    for i, query in enumerate(queries):
        print(f"\n--- Processing Query {i+1}/{len(queries)} ---")
        out_file.write(f"Query: {query}\n")
        out_file.write("="*80 + "\n")

        # 1. Retrieve passages
        retrieved_passages = retrieve_passages(query, top_k=TOP_K_RESULTS)
        out_file.write("Retrieved Context:\n\n")
        if not retrieved_passages:
             out_file.write("No passages retrieved.\n\n")
        for passage in retrieved_passages:
            out_file.write(passage.strip() + "\n\n") # Add space between passages
        out_file.write("="*80 + "\n")

        # 2. Generate answer
        if not retrieved_passages:
             generated_answer = "[Skipping generation - No context retrieved]"
        else:
             generated_answer = generate_answer(query, retrieved_passages)

        out_file.write("\nGenerated Answer:\n")
        out_file.write(generated_answer.strip() + "\n")

        out_file.write("\n" + "="*80 + "\n\n")
        print(f"--- Finished Query {i+1}/{len(queries)} ---")

        # Optional: Clear GPU cache if memory issues persist (may have limited effect with pipelines)
        # if device == 0:
        #     torch.cuda.empty_cache()

print(f"Processing complete. Results saved to {filename}")
