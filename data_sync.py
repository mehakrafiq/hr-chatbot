from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
import pickle
import re

import warnings
warnings.filterwarnings("ignore")

# Load pdfs from a directory
pdf_loader = PyPDFDirectoryLoader('Data')
pdfs = pdf_loader.load()
#pdfs

# Text Cleaning Function
for i, pdf in enumerate(pdfs):
    page = pdf.page_content
    page = re.sub(r'([\n]+)([0-9]+)', '', page)
    page = re.sub(r'([0-9]+) [.]', '', page)
    page = re.sub(r'([\n]+)', '', page)
    page = page.replace('•', '')
    page = re.sub(' +', ' ', page)
    page = page.replace("•", "")
    page = page.replace('"', "")
    if page[0].isdigit():
        page = page[1:]
    page = page.strip()
    pdfs[i].page_content = page
#pdfs

# Split the documents into smaller , managable chuncks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50
)
document_chunks = text_splitter.split_documents(pdfs)
print(f"The initial {len(pdfs)} documents were split into {len(document_chunks)} documents")

# Initialize the embedding model
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Save the embedding model configuration
embedding_config = {
    "model": "nomic-embed-text",
}

# Save the configuration to a pickle file
with open('Data/embedding_config_nomic.pkl', 'wb') as f:
    pickle.dump(embedding_config, f)

print("Embedding configuration for nomic-embed-text saved.")

# Get the embeddings for the document chunks
db = FAISS.from_documents(document_chunks, embeddings)

db.index.ntotal 


# Save the db to disk
db.save_local('Data/faiss_index')
print(f"FAISS index saved to disk. Total number of documents indexed: {db.index.ntotal}")