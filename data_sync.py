from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
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


# Get the embeddings for the document chunks
embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)


# Save the embeddings into Faiss index
db = FAISS.from_documents(document_chunks, embeddings)

db.index.ntotal 

# Save the db to disk
db.save_local('Data/faiss_index')

# Save the embedding model configuration separately (e.g., in a pickle file)
embedding_config = {
    "model_name": "BAAI/bge-small-en-v1.5",
    "model_kwargs": {'device': 'cpu'},
    "encode_kwargs": {'normalize_embeddings': True}
}
with open('Data/embedding_config.pkl', 'wb') as f:
    pickle.dump(embedding_config, f)

print("FAISS index and embedding configuration saved.")