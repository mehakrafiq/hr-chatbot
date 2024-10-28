from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings  # Updated import to align with Askari HR Chat App
from langchain_community.vectorstores import FAISS
import pickle
import re
import logging
import warnings
import os
import pandas as pd
from docx import Document
from pdfminer.high_level import extract_text

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load PDFs from a directory, extracting text only
pdf_loader = PyPDFDirectoryLoader('Docs/')

# Function to load and extract text from PDFs while ignoring images and formatting
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

pdfs = load_pdfs('Docs/')

# Function to load Word documents from a directory while ignoring images and complex formatting
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
# Update to handle encoding issues by trying different encodings

def load_csv_documents(directory):
    csv_docs = []
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            csv_path = os.path.join(directory, filename)
            try:
                df = pd.read_csv(csv_path, encoding='utf-8')
                text = df.to_string(index=False)
                csv_docs.append({"page_content": text, "metadata": {"source": filename}})
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(csv_path, encoding='ISO-8859-1')
                    text = df.to_string(index=False)
                    csv_docs.append({"page_content": text, "metadata": {"source": filename}})
                except Exception as e:
                    logging.warning(f"Could not load CSV {filename}: {e}")
            except Exception as e:
                logging.warning(f"Could not load CSV {filename}: {e}")
    return csv_docs

# Load Word and CSV documents
word_docs = load_word_documents('Docs/')
csv_docs = load_csv_documents('Docs/')

# Combine all documents
all_docs = pdfs + word_docs + csv_docs

# Refined Text Cleaning Function with Section Delimiters
def clean_text(page_content):
    page = page_content
    # Remove page numbers or specific unwanted patterns
    page = re.sub(r'Page\s*\|\s*[0-9]+', '', page, flags=re.IGNORECASE)  # Remove page markers
    # Replace multiple newlines with a section delimiter to preserve paragraph and section structure
    page = re.sub(r'\n\s*\n', '\n[SECTION_BREAK]\n', page)
    # Replace multiple spaces with a single space
    page = re.sub(' +', ' ', page)
    # Remove double quotation marks
    page = page.replace('"', '')
    # Remove leading numbers if they exist
    if page and page[0].isdigit():
        page = page[1:]
    # Remove leading and trailing whitespace
    page = page.strip()
    return page

# Apply Refined Text Cleaning Function to all documents
for i, doc in enumerate(all_docs):
    all_docs[i]['page_content'] = clean_text(doc['page_content'])

# Convert documents from dictionary format to objects with attributes to match the expected input of text_splitter
class DocumentObject:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata

converted_docs = [DocumentObject(doc['page_content'], doc['metadata']) for doc in all_docs]

# Split the documents into smaller, manageable chunks
# Adjust chunk size and overlap for optimization based on document size and expected query context
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Balanced chunk size to maintain useful context
    chunk_overlap=500  # Increased overlap for better context continuity
)
document_chunks = text_splitter.split_documents(converted_docs)
logging.info(f"The initial {len(all_docs)} documents were split into {len(document_chunks)} chunks.")

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

# Adding metadata to dynamically keep track of page numbers for later linking and redundancy information
for i, chunk in enumerate(document_chunks):
    # Extract the original document source information
    source = chunk.metadata.get("source", "unknown")
    # Dynamically assign page numbers by calculating based on chunk sequence
    # If a page number is explicitly present, use it, otherwise assign sequentially
    inferred_page_number = i + 1  # Start page numbers from 1
    page_number = chunk.metadata.get("page_number", inferred_page_number)
    chunk.metadata["page_number"] = page_number
    chunk.metadata["redundancy"] = source

    # Metadata boosting based on content type or section importance
    if "[SECTION_BREAK]" in chunk.page_content:
        if "Summary" in chunk.page_content or "Key Findings" in chunk.page_content:
            chunk.metadata["boost"] = 2  # Boost important sections for higher relevance
        else:
            chunk.metadata["boost"] = 1  # Default boost for regular sections
    else:
        chunk.metadata["boost"] = 1  # Default boost for regular content

# Save the db to disk
db.save_local('Data/faiss_index')
logging.info(f"FAISS index saved to disk. Total number of documents indexed: {db.index.ntotal}")