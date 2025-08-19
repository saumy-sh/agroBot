# Define the model path from the kaggle models imported already

########################

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

from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = "Qwen/Qwen2.5-3B-Instruct"

########################
def load_llama_model_and_tokenizer():
    # model_config = transformers.AutoConfig.from_pretrained(model_path)
    # model = transformers.AutoModelForCausalLM.from_pretrained(
    #     model_path,
    #     trust_remote_code=True,
    #     config=model_config,
    #     device_map='auto',  # Automatically distribute the model across available devices
    # )
    model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    
    return model, tokenizer

llama3_model, llama3_tokenizer = load_llama_model_and_tokenizer()



# Create the query pipeline
query_pipeline = transformers.pipeline(
    "text-generation",
    model=llama3_model,
    tokenizer=llama3_tokenizer,
    torch_dtype=torch.float16,  # Uncomment this for 4bit, faster inference but less accuracy
    max_length=2048,  # Set the maximum length of generated text
    device_map="auto",  # Automatically distribute the pipeline across available devices
    num_return_sequences=1,# Generate only one sequence
    truncation=True,
    do_sample=True,  # Enable sampling from the probability distribution
    temperature=0.2,  # Adjust the temperature for less creative responses
    pad_token_id=llama3_tokenizer.eos_token_id  # Set pad_token_id explicitly, if not you will get a warning
)

# Using the the HuggingFacePipeline class from LangChain to wrap the pipeline
def init_llm():
    llama3_llm = HuggingFacePipeline(pipeline=query_pipeline)
    return llama3_llm

