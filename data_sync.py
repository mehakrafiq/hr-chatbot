from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings  # Updated import to align with Askari HR Chat App
from langchain_community.vectorstores import FAISS
import pickle
import re
import logging

import warnings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load pdfs from a directory
pdf_loader = PyPDFDirectoryLoader('Docs/')
pdfs = pdf_loader.load()

# Text Cleaning Function
def clean_text(page_content):
    page = page_content
    page = re.sub(r'([\n]+)([0-9]+)', '', page)
    page = re.sub(r'([0-9]+) [.]', '', page)
    page = re.sub(r'([\n]+)', '', page)
    page = re.sub(' +', ' ', page)
    page = page.replace('"', '')
    if page and page[0].isdigit():
        page = page[1:]
    page = page.strip()
    return page

# Apply Text Cleaning Function
for i, pdf in enumerate(pdfs):
    pdfs[i].page_content = clean_text(pdf.page_content)

# Split the documents into smaller, manageable chunks
# Adjust chunk size and overlap for optimization based on document size and expected query context
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # Increased chunk size to balance context length
    chunk_overlap=100  # Increased overlap for better context continuity
)
document_chunks = text_splitter.split_documents(pdfs)
logging.info(f"The initial {len(pdfs)} documents were split into {len(document_chunks)} documents")

# Initialize the embedding model
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Save the embedding model configuration
embedding_config = {
    "model": "nomic-embed-text",
}

# Save the configuration to a pickle file
with open('Data/embedding_config_nomic.pkl', 'wb') as f:
    pickle.dump(embedding_config, f)

logging.info("Embedding configuration for nomic-embed-text saved.")

# Get the embeddings for the document chunks
db = FAISS.from_documents(document_chunks, embeddings)

# Adding metadata to keep track of page numbers for later linking
for doc, chunk in zip(pdfs, document_chunks):
    page_number = doc.metadata.get("page_number")
    if page_number is not None:
        chunk.metadata["page_number"] = page_number
    else:
        chunk.metadata["page_number"] = "unknown"

db.index.ntotal 

# Save the db to disk
db.save_local('Data/faiss_index')
logging.info(f"FAISS index saved to disk. Total number of documents indexed: {db.index.ntotal}")
