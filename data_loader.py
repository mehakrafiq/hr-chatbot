from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
import pdfplumber
import json
import pickle
import re
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Text Cleaning Function
def clean_text(page_content):
    page = page_content
    page = re.sub(r'([\n]+)([0-9]+)', '\n', page)
    page = re.sub(r'([0-9]+) [.]', '', page)
    page = re.sub(' +', ' ', page)
    page = page.replace('"', '').strip()
    if page and page[0].isdigit():
        page = page[1:]
    return page

# Load pdfs from a directory
pdf_loader = PyPDFDirectoryLoader('Docs/')
pdfs = pdf_loader.load()

# Extract entire PDF content and batch pages to JSON

def extract_pdf_to_json(pdf_files, output_folder="db/json_files", pages_per_batch=50):
    os.makedirs(output_folder, exist_ok=True)
    for i, pdf_file in enumerate(pdf_files):
        pdf_path = pdf_file.metadata.get('source', None)
        if pdf_path:
            with pdfplumber.open(pdf_path) as pdf:
                pdf_content = []
                for page_number, page in enumerate(pdf.pages):
                    page_text = page.extract_text() or ""
                    page_text = clean_text(page_text)
                    pdf_content.append({
                        "page_number": page_number + 1,
                        "text": page_text,
                        "type": "text"
                    })
                    tables = page.extract_tables()
                    for table in tables:
                        if table:
                            pdf_content.append({
                                "page_number": page_number + 1,
                                "table": table,
                                "type": "table"
                            })
                    # Batch pages and save them after every `pages_per_batch` pages
                    if (page_number + 1) % pages_per_batch == 0 or (page_number + 1) == len(pdf.pages):
                        batch_number = page_number // pages_per_batch
                        json_filename = os.path.join(output_folder, f"document_{i}_batch_{batch_number}.json")
                        with open(json_filename, 'w') as json_file:
                            json.dump(pdf_content, json_file, indent=4)
                        logging.info(f"Extracted content from document {i} batch {batch_number} and saved to {json_filename}")
                        pdf_content = []  # Clear content after saving

extract_pdf_to_json(pdfs)

# Load JSON files and prepare them for embedding
def load_json_files(json_folder="db/json_files"):
    json_files = [f for f in os.listdir(json_folder) if f.endswith('.json')]
    all_chunks = []
    for json_file in json_files:
        with open(os.path.join(json_folder, json_file), 'r') as f:
            data = json.load(f)
            current_page_number = 1  # Track current page number dynamically
            for item in data:
                if 'page_number' in item and item['page_number'] != "unknown":
                    current_page_number = item['page_number']
                item['page_number'] = current_page_number
                if item['type'] == 'text':
                    all_chunks.append(item)
                elif item['type'] == 'table':
                    table_text = '\n'.join(['\t'.join([str(cell) if cell is not None else '' for cell in row]) for row in item['table']])
                    all_chunks.append({
                        "page_number": item['page_number'],
                        "text": table_text,
                        "type": "table"
                    })
    return all_chunks

all_chunks = load_json_files()

# Split the documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
document_chunks = text_splitter.split_documents(all_chunks)
logging.info(f"The initial {len(all_chunks)} JSON chunks were split into {len(document_chunks)} documents")

# Initialize the embedding model
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Save the embedding model configuration
embedding_config = {"model": "nomic-embed-text"}
with open('db/embedding_config_nomic.pkl', 'wb') as f:
    pickle.dump(embedding_config, f)

logging.info("Embedding configuration for nomic-embed-text saved.")

# Get the embeddings for the document chunks
db = FAISS.from_documents(document_chunks, embeddings)

# Add metadata for page numbers
for chunk in document_chunks:
    chunk["page_number"] = chunk.get("page_number", "unknown")

db.index.ntotal 

# Save the db to disk
db.save_local('db/faiss_index')
logging.info(f"FAISS index saved to disk. Total number of documents indexed: {db.index.ntotal}")
