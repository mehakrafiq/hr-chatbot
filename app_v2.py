import streamlit as st
from streamlit_chat import message
from langchain_community.vectorstores import FAISS  # Updated import for FAISS
from langchain_ollama import OllamaLLM  # Updated import for Ollama
from langchain.chains.llm import LLMChain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.embeddings import OllamaEmbeddings
import pickle
import os


# Load and display the uploaded image at the top of the page

st.set_page_config(layout="wide")


logo_path = 'Image/digitallogo.jpg'
with st.sidebar:
    st.image(logo_path, width=150)
    st.markdown("### Askari Digital HR Assistant")
    st.markdown("Welcome to Askari Bank Personal Assistant - developed by DBD team.")



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
embedding_config_path = 'Data/embedding_config_nomic.pkl'

@st.cache_resource
def load_embeddings():
    with open(embedding_config_path, 'rb') as f:
        embedding_config = pickle.load(f)
    embeddings = OllamaEmbeddings(
        model=embedding_config["model"]
    )
    return embeddings

@st.cache_resource
def load_faiss_index(_embeddings):
    if os.path.exists(faiss_index_path):
        db = FAISS.load_local(
            folder_path=faiss_index_path,
            embeddings=_embeddings,
            allow_dangerous_deserialization=True
        )
        return db
    else:
        st.error("FAISS index not found. Please check the path.")
        return None

@st.cache_resource
def llm_pipeline():
    return OllamaLLM(model="llama3.2")

@st.cache_resource
def qa_llm():
    embeddings = load_embeddings()
    db = load_faiss_index(embeddings)
    if db is not None:
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        chat_prompt = ChatPromptTemplate.from_template(prompt_template)
        llm = llm_pipeline()
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": chat_prompt}
        )
        return qa
    else:
        return None

# Display conversation history using Streamlit messages
def display_conversation(history):
    for i in range(len(history["generated"])):
        if i < len(history["past"]):
            st.chat_message("user", avatar="🧑‍💼").markdown(history["past"][i])
        if i < len(history["generated"]):
            st.chat_message("assistant", avatar="👩🏻‍💻").markdown(history["generated"][i])


def process_answer(query):
    qa = qa_llm()
    if qa is not None:
        try:
            answer = qa.invoke({"query": query})
            return answer['result']
        except AssertionError as e:
            st.error(f"AssertionError: {str(e)}")
            return "An error occurred due to a dimensionality mismatch. Please check the FAISS index and embedding configurations."
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")
            return "An unexpected error occurred. Please try again later."
    else:
        return "I'm unable to retrieve an answer at the moment."


def submit_input():
    st.session_state['input_submitted'] = True

def main():
    st.session_state.submit_input = submit_input
    st.write("Welcome to Askari Bank Personal Assistant")

    # Initialize session state for generated responses and past messages
    if "generated" not in st.session_state:
        st.session_state["generated"] = []
    if "past" not in st.session_state:
        st.session_state["past"] = []

    # Display conversation history first
    if st.session_state["generated"]:
        display_conversation(st.session_state)

    # User input for query below the conversation history
    user_query = st.chat_input("Enter your HR-related question:")

    if user_query:
        st.session_state['input_submitted'] = False
        response = process_answer(user_query)

        # Update session state
        st.session_state["past"].append(user_query)
        st.session_state["generated"].append(response)

        # Re-display conversation after getting a new response
        display_conversation(st.session_state)

if __name__ == '__main__':
    main()
