import streamlit as st
from langchain_community.vectorstores import FAISS  # Updated import for FAISS
from langchain_ollama import OllamaLLM  # Updated import for Ollama
from langchain.chains.llm import LLMChain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import pickle
import os


#st.set_page_config(layout="wide")

# Load and display the uploaded image at the top of the page
logo_path = 'Image/digitallogo.jpg'
st.image(logo_path, width=150)  # Adjust the width as per your requirement

# Set up Streamlit interface
st.title("Askari HR Assistant")

# Define the prompt template
prompt_template = """
You are a friendly senior Human Resource(HR) personnel. 
You will be given a question from an employee regarding their queries related to HR-Policies. 
Your Task is to understand the employee's question first thoroughly then based on the context provided to you, 
you need to answer to the employee. Make sure your answer conveys a clear, concise, and limited to 3-4 sentences 
response to the user question. If you don't know the answer to employee's question, you will say "I don't know the answer to your question, 
Please contact the focal HR personnel in your department."

Here is how you will operate:

Context: {context}
Question: {question}
Your Helpful Answer:  """

# Define the path to the FAISS index and the embedding config file
faiss_index_path = 'Data/faiss_index'
embedding_config_path = 'Data/embedding_config.pkl'

# Load embedding configuration from the saved pickle file
if os.path.exists(embedding_config_path):
    with open(embedding_config_path, 'rb') as f:
        embedding_config = pickle.load(f)

    # Initialize the embeddings based on the saved configuration
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=embedding_config["model_name"],
        model_kwargs=embedding_config["model_kwargs"],
        encode_kwargs=embedding_config["encode_kwargs"]
    )

    # Load the FAISS index from disk
    if os.path.exists(faiss_index_path):
        db = FAISS.load_local(
            folder_path=faiss_index_path,
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
        st.write(f"Welcome to Askari Bank Personal Assistant")

        # Create a retriever with FAISS
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

        # Create the chat prompt
        chat_prompt = ChatPromptTemplate.from_template(prompt_template)

        # Initialize the OllamaLLM model
        model = OllamaLLM(model="llama3.2")  # Updated to use OllamaLLM

        # Set up the RetrievalQA chain
        retrievalQA = RetrievalQA.from_chain_type(
            llm=model,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": chat_prompt}
        )

        # User input for query
        user_query = st.text_input("Enter your HR-related question:")

        if st.button("Get Answer"):
            # Get the answer from the retrieval QA chain
            answer = retrievalQA.invoke({"query": user_query})

            # Display the query and the result
            st.write("Query:", user_query)
            st.write("Answer:", answer['result'])

            # # Display the source documents' metadata (e.g., page numbers)
            # pages = [doc.metadata.get('page', 'N/A') for doc in answer['source_documents']]
            # pages.sort()
            # st.write(f"Further information can be found on pages: {pages}")
    else:
        st.error("FAISS index not found. Please check the path.")
else:
    st.error("Embedding configuration not found. Please check the path to 'embedding_config.pkl'.")
