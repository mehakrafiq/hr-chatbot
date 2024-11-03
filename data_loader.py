import os
import re
import pickle
import logging
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings  
from langchain_community.vectorstores import FAISS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load TXT files from a directory
def load_txt_documents(directory):
    txt_docs = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            txt_path = os.path.join(directory, filename)
            try:
                with open(txt_path, 'r', encoding='utf-8') as file:
                    text = file.read()
                txt_docs.append({"page_content": text, "metadata": {"source": filename}})
            except Exception as e:
                logging.warning(f"Could not load TXT {filename}: {e}")
    return txt_docs

# Load CSV files from a directory
def load_csv_documents(directory):
    csv_docs = []
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            csv_path = os.path.join(directory, filename)
            try:
                df = pd.read_csv(csv_path, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(csv_path, encoding='ISO-8859-1')
            except Exception as e:
                logging.warning(f"Could not load CSV {filename}: {e}")
                continue
            text = df.to_string(index=False)
            csv_docs.append({"page_content": text, "metadata": {"source": filename}})
    return csv_docs

# Load documents
txt_docs = load_txt_documents('Docs/')
csv_docs = load_csv_documents('Docs/')
all_docs = txt_docs + csv_docs

logging.info(f"Loaded {len(txt_docs)} TXT documents.")
logging.info(f"Loaded {len(csv_docs)} CSV documents.")
logging.info(f"Loaded a total of {len(all_docs)} documents.")

# Refined Text Cleaning Function
def clean_text(page_content):
    # Remove page markers, multiple newlines, extra spaces, section breaks, and other unnecessary characters
    page = re.sub(r'Page\s*\|\s*[0-9]+', '', page_content, flags=re.IGNORECASE)  # Remove page markers
    page = re.sub(r'\n+', ' ', page)  # Replace multiple newlines with a single space
    page = re.sub(r'-{2,}', ' ', page)  # Replace section breaks or repeated dashes with a space
    page = re.sub(r'\s+', ' ', page)  # Replace multiple spaces with a single space
    page = re.sub(r'[\r\f\v]', '', page)  # Remove carriage returns, form feeds, vertical tabs
    page = re.sub(' +', ' ', page).replace('"', '').strip()  # Clean up remaining whitespace and quotes
    return page[1:] if page and page[0].isdigit() else page

# Clean all documents
for doc in all_docs:
    doc['page_content'] = clean_text(doc['page_content'])

# Convert documents to objects with attributes
class DocumentObject:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata

converted_docs = [DocumentObject(doc['page_content'], doc['metadata']) for doc in all_docs]

# Split documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
document_chunks = text_splitter.split_documents(converted_docs)
logging.info(f"The initial {len(all_docs)} documents were split into {len(document_chunks)} chunks.")

# Initialize embeddings and save configuration
embeddings = OllamaEmbeddings(model="nomic-embed-text")
with open('Data/embedding_config_nomic.pkl', 'wb') as f:
    pickle.dump({"model": "nomic-embed-text"}, f)
logging.info("Embedding configuration saved.")

# Get embeddings and save FAISS index
db = FAISS.from_documents(document_chunks, embeddings)
db.save_local('Data/faiss_index')
logging.info(f"FAISS index saved. Total number of documents indexed: {db.index.ntotal}")
