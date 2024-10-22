from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings  # Updated import to align with Askari HR Chat App
from langchain_community.vectorstores import FAISS
import pdfplumber
import json
import pickle
import re
import logging
import os

import warnings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load pdfs from a directory
pdf_loader = PyPDFDirectoryLoader('Docs/')
pdfs = pdf_loader.load()

# Extract tables and save to JSON
def extract_tables_to_json(pdf_files, output_folder="db/json_tables"):
    os.makedirs(output_folder, exist_ok=True)
    for i, pdf_file in enumerate(pdf_files):
        pdf_path = pdf_file.metadata.get('source', None)
        if pdf_path:
            with pdfplumber.open(pdf_path) as pdf:
                all_tables = []
                for page_number, page in enumerate(pdf.pages):
                    tables = page.extract_tables()
                    for table in tables:
                        all_tables.append({
                            "page_number": page_number + 1,
                            "table": table,
                            "type": "table"  # Adding metadata to identify as table
                        })
                # Save tables to a JSON file
                if all_tables:
                    json_filename = os.path.join(output_folder, f"document_{i}_tables.json")
                    with open(json_filename, 'w') as json_file:
                        json.dump(all_tables, json_file)
                    logging.info(f"Extracted tables from document {i} and saved to {json_filename}")

# Call the function to extract tables
extract_tables_to_json(pdfs)

# Text Cleaning Function (for non-tabular data)
def clean_text(page_content):
    page = page_content
    page = re.sub(r'([\n]+)([0-9]+)', '', page)
    page = re.sub(r'([0-9]+) [.]', '', page)
    # Retain newlines for better structure preservation
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
with open('db/embedding_config_nomic.pkl', 'wb') as f:
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
db.save_local('db/faiss_index')
logging.info(f"FAISS index saved to disk. Total number of documents indexed: {db.index.ntotal}")
