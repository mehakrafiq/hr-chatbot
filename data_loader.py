from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings  
from langchain_community.vectorstores import FAISS
import pickle
import logging
import os
import re
import pandas as pd
from docx import Document
from pdfminer.high_level import extract_text

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load PDFs from a directory, extracting text only
def load_pdfs(directory):
    pdf_docs = []
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(directory, filename)
            try:
                text = extract_text(pdf_path)
                pdf_docs.append({"page_content": text, "metadata": {"source": filename}})
            except Exception as e:
                logging.warning(f"Could not extract text from {filename}: {e}")
    return pdf_docs

# Load Word documents from a directory
def load_word_documents(directory):
    word_docs = []
    for filename in os.listdir(directory):
        if filename.endswith(".docx"):
            doc_path = os.path.join(directory, filename)
            try:
                doc = Document(doc_path)
                text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
                word_docs.append({"page_content": text, "metadata": {"source": filename}})
            except Exception as e:
                logging.warning(f"Could not extract text from {filename}: {e}")
    return word_docs

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
pdfs = load_pdfs('Docs/')
word_docs = load_word_documents('Docs/')
csv_docs = load_csv_documents('Docs/')
all_docs = pdfs + word_docs + csv_docs

logging.info(f"Loaded {len(pdfs)} PDF documents.")
logging.info(f"Loaded {len(word_docs)} Word documents.")
logging.info(f"Loaded {len(csv_docs)} CSV documents.")
logging.info(f"Loaded a total of {len(all_docs)} documents.")

# Refined Text Cleaning Function
def clean_text(page_content):
    page = re.sub(r'Page\s*\|\s*[0-9]+', '', page_content, flags=re.IGNORECASE)  # Remove page markers
    page = re.sub(r'\n\s*\n', '\n\n', page)  # Replace multiple newlines with two newlines (user-friendly paragraphing)
    page = re.sub(' +', ' ', page).replace('"', '').strip()  # Clean up whitespace and quotes
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
