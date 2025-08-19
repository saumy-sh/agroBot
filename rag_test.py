CHROMA_PATH = "./chroma"
DATA_PATH = "./database/rag.pdf"


import torch
import langchain
import os
from pypdf import PdfReader
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain.vectorstores.chroma import Chroma
from langchain_community.llms import Ollama
from langchain.llms import HuggingFacePipeline
import numpy as np
import re
import shutil
import transformers
from transformers import AutoModel, AutoTokenizer
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
import warnings
from FlagEmbedding import FlagReranker
from transformers import AutoModel, AutoTokenizer, T5ForConditionalGeneration
from tqdm import tqdm

# importing model
from llm_test import init_llm

llama3_llm = init_llm()




def clean_text(page_content):
    # Step 1: Remove excessive whitespace, tabs, and newlines
    cleaned_text = re.sub(r'\s+', ' ', page_content)

    # Step 2: Remove URLs or links
    cleaned_text = re.sub(r'http[s]?://\S+|www\.\S+', '', cleaned_text)

    # Step 3: Remove unwanted special characters (keep basic punctuation)
    cleaned_text = re.sub(r'[^A-Za-z0-9\s,.!?\'"-]', '', cleaned_text)

    # Step 4: Strip leading/trailing whitespace
    cleaned_text = cleaned_text.strip()
    
    return cleaned_text


def load_documents():
    '''
        Load the documents ,
        clean the text and save the clean text in the document's page content.

        Returns: documents

    '''
    document_loader = PyPDFLoader(DATA_PATH)

    documents = document_loader.load()

    for doc in documents:
        if doc.metadata["page"] >= 645:
            doc.page_content = " "
        else:
            doc.page_content = clean_text(doc.page_content)
    return documents

def split_documents(documents):

    """
        Using simple Recursive Text Splitter we divide the document content into chunks.

        Returns: split_documents
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
   
    return text_splitter.split_documents(documents)


device = "cpu" 
model_kwargs = {"device":device}


# Set up the embeddings for the Meta LLama 3.2 1-b instruct version.
embedding_model_name = "BAAI/bge-large-en-v1.5"
# encoder = SentenceTransformer(embedding_model_name, device)
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name, model_kwargs=model_kwargs, 
                                  encode_kwargs={'normalize_embeddings':True})



def calculate_chunk_ids_and_keyphrases(chunks):

    # This will create IDs like "path_to_book:6:2"
    # The ID has the following format: Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in tqdm(chunks, desc = "Calculating IDs:"):
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page - 12}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

        # Tried to extract the main keyphrases for each chunk content, but increases the inference time by a lot

    return chunks

def add_to_chroma(chunks):
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=embeddings
    )

    chunks_with_ids = calculate_chunk_ids_and_keyphrases(chunks)
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])

    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
    else:
        print("✅ No new documents to add")
    return db

def create_database():

    documents = load_documents()
    chunks = split_documents(documents)
    db = add_to_chroma(chunks)
    
    return db
    
def clear_database():
    
    '''
        Clear the database if need a fresh start.
    '''
    
    if os.path.exists(CHROMA_PATH):
        print("True")
        shutil.rmtree(CHROMA_PATH)
# chroma_db = create_database()


# Loading the Reranking Model (Also open-sourced from Huggingface) 
reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True, device=device) 


def retrieve_reranked_documents(query, chroma_db, number_of_docs, reranker):

    """
        Returns the re-ranked documents from the vector store
    """
    results = chroma_db.similarity_search(query, k = number_of_docs * 10)

    pairs = [[query, result.page_content] for result in results]
    scores = reranker.compute_score(pairs, normalize=True)

    sorted_results = sorted(zip(results, scores), key = lambda x : x[1], reverse=True)

    retrieved_documents = [(result.page_content, result.metadata["id"], score) for result, score in sorted_results[:number_of_docs]]

    return retrieved_documents




# Creating mappings for references 
import pymupdf # PyMuPDF

def extract_toc(pdf_path):
    # Open the PDF
    doc = pymupdf.open(pdf_path)
    # Get the Table of Contents
    toc = doc.get_toc()
    # Each entry is a list: [level, title, page]
    for content in toc[2:]:
        content[2] = content[2] - 12
    return toc

def create_mapping(toc):
    
    '''
    Create a mapping between page numbers and sections for references.
    '''

    mapping = {}
    current_section = None

    for entry in toc:
        level, title, page = entry
        if level == 1:  # It's a main section
            current_section = title
            mapping[current_section] = {}
        elif level == 2 and current_section:  # It's a subsection
            if title not in ["Introduction", "Critical Thinking Questions", "Personal Application Questions", "Review Questions"]:
                mapping[current_section][title] = {"start_page": page, "end_page": None}
        elif level == 3:  # Handle deeper levels if needed
            continue  # Optional: Extend this logic for deeper subsections

    # Update end pages for subsections
    all_entries = list(toc)  # Flatten the TOC for easier traversal
    for i in range(len(all_entries) - 1):
        level, title, page = all_entries[i]
        next_level, next_title, next_page = all_entries[i + 1]

        # Only update if it's a subsection or section
        if level >= 1:
            current_section = None
            for section, subsections in mapping.items():
                if title == "Contents":
                    continue
                if title in subsections:
                    current_section = section
                    break
            if current_section and (title == "Summary" or title == "Key Terms"):
                mapping[current_section][title]["end_page"] = mapping[current_section][title]["start_page"] + 1
            
            elif current_section:
                mapping[current_section][title]["end_page"] = next_page
    # Update last subsections to stretch to the end of the document
    for section, subsections in mapping.items():
        for subsection, pages in subsections.items():
            if pages["end_page"] is None:
                pages["end_page"] = pages["start_page"] + 1 # Assign infinite if no end page exists

    return mapping

toc = extract_toc("/kaggle/input/rag-dataset/rag.pdf")
mapping = create_mapping(toc)



# Function to query RAG and get the response 

def query_rag(query, chroma_db, number_of_docs):

    transformed_query = generate_hypothetical_response(query)
    reranked_documents = retrieve_reranked_documents(transformed_query, chroma_db, 
                                                     number_of_docs,
                                                    reranker)
    
    final_documents = [doc for doc,  _ , _ in reranked_documents]

    pages = [page.split(":")[1] for _, page, _ in reranked_documents]
    context_text = "\n\n".join(final_documents)


    # Prompt template for the LLM
    messages = f""" You are an AI Assistant expert in crop, agriculture science. Your job is to read the crop disease and suggest a proper treatment with precautionary measures.
    This response is going to be used by a local native farmer so keep the information digestable and concise. 
        
        Context: {context_text}
    
        Instructions:
        1. Understand all the context provided to craft your response and ensure relevance.
        2. Avoid unnecessary repetition in your response.
        4. Exclude closing phrases such as "Best regards" or "Let me know if I can help you further."
        
        Task: Respond to the following question by providing a detailed and relevant answer keeping in mind the context.

        Question: {query}
        
        Answer:

    """
    
    response_text = llama3_llm.invoke(input=messages)
    
    prompt = messages.split("Answer:")[0].strip()
    
    answer = response_text.split("Answer:")[1].strip()
    
    final_output_llm = f"{answer}"
    
    # Return the context, response, and page numbers
    return context_text, final_output_llm, pages


def resolve_references(mapping, pages):
    
    """
    Resolves the references for given pages in the format 
    "section/subsection" for better reference stucture.
    """
    
    references = []

    for page in pages:
        for section, subsections in mapping.items():
            for subsection, page_range in subsections.items():

                if page_range["start_page"] is None:
                    continue
                if page == page_range["start_page"]:
                    references.append(f"{section}/{subsection}")
                if page == page_range["end_page"]:
                    references.append(f"{section}/{subsection}")
                if page_range["start_page"] <= int(page) <= page_range["end_page"]:
                    references.append(f"{section}/{subsection}")
                    break  # Stop searching once matched
            else:
                continue
            break

    return references



import json
def answer_queries(query):
    """
        This function queries the RAG for each query in the json file and saves it in the csv file.

        Params: json file path

        Returns: CSV file
    """
    # with open(json_file) as f:
    #     queries = json.load(f)
    
    # print(f"Answering question \n\n "
    context_text, response_text, pages = query_rag(query, chroma_db, 8)
    references = resolve_references(mapping, pages)
    references = {"sections": list(set(references)), "pages": [f"{page}" for page in pages]}

    # print(f"context: {context_text}")
    # print(f"Answer from LLM: {response_text}")
    # print(f"References: {references}")
    return response_text




