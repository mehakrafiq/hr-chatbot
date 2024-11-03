from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
import pickle
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)  # Enable CORS to handle cross-origin requests

# Load necessary configurations and models
faiss_index_path = 'Data/faiss_index'
embedding_config_path = 'Data/embedding_config_nomic.pkl'


def load_embeddings():
    with open(embedding_config_path, 'rb') as f:
        embedding_config = pickle.load(f)
    embeddings = OllamaEmbeddings(model=embedding_config["model"])
    return embeddings


def load_faiss_index(_embeddings):
    if os.path.exists(faiss_index_path):
        db = FAISS.load_local(
            folder_path=faiss_index_path,
            embeddings=_embeddings,
            allow_dangerous_deserialization=True
        )
        return db
    else:
        return None


def llm_pipeline():
    return OllamaLLM(model="biggermistral")


def qa_llm():
    embeddings = load_embeddings()
    db = load_faiss_index(embeddings)
    if db is not None:
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5, "scoring_function": custom_scoring_function})
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


prompt_template = """
You are a friendly senior Human Resource(HR) personnel. 
You will be given a question from an employee regarding their queries related to HR-Policies. 
Your Task is to understand the employee's question first thoroughly then based on the context provided to you, 
you need to answer to the employee. Make sure your answer conveys a clear, concise, and limited to 3-4 sentences 
response to the user question. 
If you don't know the answer to employee's question, you will say "I don't know the answer to your question, 
Please contact the focal HR personnel in your department."

Context: {context}
Question: {question}
Your Helpful Answer:  """

# Define a custom scoring function to leverage the "boost" metadata
def custom_scoring_function(chunk):
    base_score = chunk.similarity_score  # Use similarity score as a base
    boost_factor = chunk.metadata.get("boost", 1)
    # Prioritize CSV content for factual responses
    if chunk.metadata.get("content_type") == "CSV":
        boost_factor *= 2
    return base_score * boost_factor



@app.route('/')
def index():
    return render_template('index.html')  # Assuming index.html is your frontend


@app.route('/ask_hr', methods=['POST'])
def ask_hr():
    data = request.get_json()  # Get JSON data from the request
    user_query = data.get('query') if data else None
    #print(f"Received API request with query: {user_query}")  # Debugging statement
    if user_query and user_query.strip():
        response = process_answer(user_query)
        #print(f"Response generated: {response}")  # Debugging line
        return jsonify({'response': response})
    else:
        return jsonify({'response': "Please provide a valid HR-related question."})


def process_answer(query):
    qa = qa_llm()
    if qa is not None:
        try:
            # Build the context including the conversation history
            context = f"Relevant HR context here."
            prompt = prompt_template.format(context=context, question=query)
            #print(f"Generated prompt: {prompt}")  # Debugging line
            answer = qa.invoke(input=prompt)
            response_text = answer['result']
            #print(f"Model answer: {response_text}")  # Debugging line
            return response_text
        except AssertionError as e:
            print(f"AssertionError: {e}")  # Debugging line
            return "An error occurred due to a dimensionality mismatch. Please check the FAISS index and embedding configurations."
        except Exception as e:
            print(f"Unexpected error: {e}")  # Debugging line
            return "An unexpected error occurred. Please try again later."
    else:
        print("QA object not initialized properly")  # Debugging line
        return "I'm unable to retrieve an answer at the moment."


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5050)  # Configurable as needed
