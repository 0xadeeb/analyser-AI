# Reference -> https://github.com/mrdbourke/simple-local-rag

import os
import fitz
import time
import textwrap
import torch
import numpy as np 
import pandas as pd
import re

from tqdm.auto import tqdm # for progress bars, requires !pip install tqdm 
from spacy.lang.en import English
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import util, SentenceTransformer

gpu_dev = "mps"

# Get PDF document
pdf_path = "../data/IOS Architecture.pdf"

def func_timer(f):
    def timed(*args, **kwargs):
        ts = time.time()
        result = f(*args, **kwargs)
        te = time.time()
        print(f"[INFO] Time taken to exceput fn {f.__name__}: {te-ts:.5f} seconds.")
        return result
    return timed

def text_formatter(text: str) -> str:
    """Performs minor formatting on text."""
    cleaned_text = text.replace("\n", " ").strip() # note: this might be different for each doc (best to experiment)

    # Other potential text formatting functions can go here
    return cleaned_text

# Open PDF and get lines/pages
# Note: this only focuses on text, rather than images/figures etc
def open_and_read_pdf(pdf_path: str) -> list[dict]:
    if not os.path.exists(pdf_path):
        print("File doesn't exist")
    else:
        print(f"File {pdf_path} exists.")

    doc = fitz.open(pdf_path)  # open a document
    pages_and_texts = []
    nlp = English()
    nlp.add_pipe("sentencizer")

    for page_number, page in tqdm(enumerate(doc)):
        text = page.get_text()
        text = text_formatter(text)
        sentences = list(map(str, (nlp(text).sents)))
        pages_and_texts.append({
                "page_number": page_number,
                "page_char_count": len(text),
                "page_word_count": len(text.split(" ")),
                "page_token_count": len(text) / 4,  # 1 token = ~4 chars, see: https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
                "text": text,
                "sentences": sentences,
                "page_sentence_count": len(sentences)
            }
        )

    return pages_and_texts

# Create a function that recursively splits a list into desired sizes
def split_list(input_list: list, token_capacity: int, tolerance=0.9) -> list[list[str]]:
    chunked_list = []
    chunk = []
    curr_size = 0
    
    # Instead of a loop, the some formula like
    # sent_per_chunk = token_cap × μt/μs × (1−μt/σt ) × (1−α)
    # can be used if standard deviation of token and sents per page is less
    for sent in input_list:
        if curr_size + len(sent) <= 4 * tolerance * token_capacity: # One token = ~4 chars
            chunk.append(sent)
            curr_size += len(sent)
        else:
            chunked_list.append(chunk)
            chunk = [sent]
            curr_size = len(sent)

    chunked_list.append(chunk)
    return chunked_list

def text_to_sent_chucks(pages_and_texts, token_capacity): 
    # Loop through pages and texts and split sentences into chunks
    for item in tqdm(pages_and_texts):
        item["sentence_chunks"] = split_list(input_list=item["sentences"], token_capacity=token_capacity)
        item["num_chunks"] = len(item["sentence_chunks"])

    pages_and_chunks = []
    for item in tqdm(pages_and_texts):
        for sentence_chunk in item["sentence_chunks"]:
            chunk_dict = {}
            chunk_dict["page_number"] = item["page_number"]
            
            # Join the sentences together into a paragraph-like structure, aka a chunk (so they are a single string)
            joined_sentence_chunk = "".join(sentence_chunk).replace("  ", " ").strip()
            joined_sentence_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_sentence_chunk)
            chunk_dict["sentence_chunk"] = joined_sentence_chunk

            # Get stats about the chunk
            chunk_dict["chunk_char_count"] = len(joined_sentence_chunk)
            chunk_dict["chunk_word_count"] = len([word for word in joined_sentence_chunk.split(" ")])
            chunk_dict["chunk_token_count"] = len(joined_sentence_chunk) / 4 # 1 token = ~4 characters
            
            pages_and_chunks.append(chunk_dict)

    return pages_and_chunks

@func_timer
def vectorize_chunks(pages_and_chunks_over_min_token_len, path):
    embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2", device=gpu_dev)
    # Create embeddings one by one on the GPU
    for item in tqdm(pages_and_chunks_over_min_token_len):
        item["embedding"] = embedding_model.encode(item["sentence_chunk"])

    # Save embeddings to file
    text_chunks_and_embeddings_df = pd.DataFrame(pages_and_chunks_over_min_token_len)
    text_chunks_and_embeddings_df.to_csv(path, index=False)


def get_embedding_df(path):
    text_chunks_and_embedding_df = pd.read_csv(path)
    
    text_chunks_and_embedding_df["embedding"] = text_chunks_and_embedding_df["embedding"].apply(
        lambda x: np.fromstring(x.strip("[]"), sep=" ")
    )
    
    return text_chunks_and_embedding_df

# Example tensors
@func_timer
def retrieve_relevant_resources(query: str,
                                embeddings: torch.tensor,
                                model: SentenceTransformer,
                                n_resources_to_return: int=5):
    """
    Embeds a query with model and returns top k scores and indices from embeddings.
    """

    # Embed the query
    query_embedding = model.encode(query, 
                                   convert_to_tensor=True) 

    # Get dot product scores on embeddings
    dot_scores = util.dot_score(query_embedding, embeddings)[0]

    scores, indices = torch.topk(input=dot_scores, k=n_resources_to_return)

    return scores, indices

# Define helper function to print wrapped text 
def print_wrapped(text, wrap_length=80):
    wrapped_text = textwrap.fill(text, wrap_length)
    print(wrapped_text)

def print_top_results_and_scores(query: str,
                                 embeddings: torch.tensor,
                                 pages_and_chunks: list[dict],
                                 n_resources_to_return: int=5):
    """
    Takes a query, retrieves most relevant resources and prints them out in descending order.

    Note: Requires pages_and_chunks to be formatted in a specific way (see above for reference).
    """
    
    scores, indices = retrieve_relevant_resources(query=query,
                                                  embeddings=embeddings,
                                                  n_resources_to_return=n_resources_to_return)
    
    print(f"Query: {query}\n")
    print("Results:")
    # Loop through zipped together scores and indicies
    for score, index in zip(scores, indices):
        print(f"Score: {score:.4f}")
        # Print relevant sentence chunk (since the scores are in descending order, the most relevant chunk will be first)
        print_wrapped(pages_and_chunks[index]["sentence_chunk"])
        # Print the page number too so we can reference the textbook further and check the results
        print(f"Page number: {pages_and_chunks[index]['page_number']}")
        print("\n")


def get_model_num_params(model: torch.nn.Module):
    return sum([param.numel() for param in model.parameters()])

def get_model_mem_size(model: torch.nn.Module):
    """
    Get how much memory a PyTorch model takes up.
    """
    # Get model parameters and buffer sizes
    mem_params = sum([param.nelement() * param.element_size() for param in model.parameters()])
    mem_buffers = sum([buf.nelement() * buf.element_size() for buf in model.buffers()])

    # Calculate various model sizes
    model_mem_bytes = mem_params + mem_buffers # in bytes
    model_mem_mb = model_mem_bytes / (2**20) # in megabytes
    model_mem_gb = model_mem_bytes / (2**30) # in gigabytes

    return {"model_mem_bytes": model_mem_bytes,
            "model_mem_mb": round(model_mem_mb, 2),
            "model_mem_gb": round(model_mem_gb, 2)}

def prompt_formatter(query: str, pages_and_chunks, indices, scores, tokenizer) -> str:
    """
    Augments query with text-based context from context_items.
    """
    # Create a list of context items
    context_items = [pages_and_chunks[i] for i in indices]

    # Add score to context item
    for i, item in enumerate(context_items):
        item["score"] = scores[i].cpu() # return score back to CPU 
        
    # Join context items into one dotted paragraph
    context = "- " + "\n- ".join([item["sentence_chunk"] for item in context_items])

    # Create a base prompt with examples to help the model
    # Note: this is very customizable, I've chosen to use 3 examples of the answer style we'd like.
    # We could also write this in a txt file and import it in if we wanted.
    base_prompt = """Based on the following context items, please answer the query.
Give yourself room to think by extracting relevant passages from the context before answering the query.
Don't return the thinking, only return the answer.
Make sure your answers are as explanatory as possible.
\nNow use the following context items to answer the user query:
{context}
\nRelevant passages: <extract relevant passages from the context here>
User query: {query}
Answer:"""

    # Update base prompt with context items and query   
    base_prompt = base_prompt.format(context=context, query=query)

    # Create prompt template for instruction-tuned model
    dialogue_template = [
        {"role": "user",
        "content": base_prompt}
    ]
    print("Question: ", base_prompt)
    # Apply the chat template
    prompt = tokenizer.apply_chat_template(conversation=dialogue_template,
                                          tokenize=False,
                                          add_generation_prompt=True)
    return prompt

@func_timer
def generate_answer(prompt, tokenizer, model):
    input_ids = tokenizer(prompt, return_tensors="pt").to(gpu_dev)

    outputs = model.generate(**input_ids,
                                 temperature=0.7,
                                 do_sample=True,
                                 max_new_tokens=512)
    
    # Turn the output tokens into text
    output_text = tokenizer.decode(outputs[0])
    return output_text.replace(prompt, "").replace("<bos>", "").replace("<eos>", "").replace("Sure, here is the answer to the user query:\n\n", "")

def main() -> None:
    embedding_path = "../data/IOS-Architecture.csv"
    if not os.path.exists(embedding_path):
        pages_and_texts = open_and_read_pdf(pdf_path=pdf_path)
        df = pd.DataFrame(pages_and_texts)

        embedding_model_token_cap = 370
        pages_and_chunks = text_to_sent_chucks(pages_and_texts, embedding_model_token_cap)

        df = pd.DataFrame(pages_and_chunks)

        # Show random chunks with under 30 tokens in length
        min_token_length = 30

        # Remove chunks where token size is less than min_token_length
        df = df[df["chunk_token_count"] > min_token_length].to_dict(orient="records")

        vectorize_chunks(df, embedding_path)

    text_chunks_and_embedding_df = get_embedding_df(embedding_path)
    pages_and_chunks = text_chunks_and_embedding_df.to_dict(orient="records")

    # Convert embeddings to torch tensor and send to device (note: NumPy arrays are float64, torch tensors are float32 by default)
    embeddings = torch.tensor(np.array(text_chunks_and_embedding_df["embedding"].tolist()), dtype=torch.float32).to(gpu_dev)
    embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2", device=gpu_dev)

    use_quantization_config = False
    model_id = "google/gemma-2b-it"

    # Not used now
    quantization_config = BitsAndBytesConfig(load_in_4bit=True,
                                            bnb_4bit_compute_dtype=torch.float16)
    # Setup Flash attention
    attn_implementation = "sdpa"
    print(f"[INFO] Using attention implementation: {attn_implementation}")
    print(f"[INFO] Using model_id: {model_id}")

    # 3. Instantiate tokenizer (tokenizer turns text into numbers ready for the model) 
    access_token = os.environ['HUGGING_FACE_TOKEN']
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_id, token=access_token,
                                            quantization_config=quantization_config)

    llm_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_id, 
                                                    torch_dtype=torch.float16, # datatype to use, we want float16
                                                    quantization_config=quantization_config if use_quantization_config else None,
                                                    low_cpu_mem_usage=False, # use full memory 
                                                    attn_implementation=attn_implementation,
                                                    token=access_token) # which attention version to use

    if not use_quantization_config: # quantization takes care of device setting automatically, so if it's not used, send model to GPU 
        llm_model.to(gpu_dev)

    print("Model size: ", get_model_mem_size(llm_model))

    query = input("Enter query: ")

    # RETRIVAL
    scores, indices = retrieve_relevant_resources(query=query, embeddings=embeddings, model=embedding_model)
    
    # AUGUMENT
    prompt = prompt_formatter(query, pages_and_chunks, indices, scores, tokenizer)
    
    # GENERATE
    answer = generate_answer(prompt=prompt, tokenizer=tokenizer, model=llm_model)

    print(f"Answer:\n")
    print_wrapped(answer)
    print(f"Context items:")
    # print(context_items)

main()