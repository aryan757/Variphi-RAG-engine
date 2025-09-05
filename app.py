import os
import base64
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Logo helper functions
def get_logo_base64(logo_path="download-removebg-preview.png"):
    """Get base64 encoded logo data"""
    try:
        with open(logo_path, "rb") as logo_file:
            logo_data = logo_file.read()
            return base64.b64encode(logo_data).decode()
    except FileNotFoundError:
        return None
    except Exception as e:
        st.error(f"Error loading logo: {str(e)}")
        return None

def display_logo(logo_path="download-removebg-preview.png", width=80, height=80):
    """Display company logo from file with black background"""
    logo_base64 = get_logo_base64(logo_path)
    if logo_base64:
        st.markdown(f"""
        <div style="text-align: center; margin-bottom: 20px; background-color: #000000; padding: 15px; border-radius: 10px; display: inline-block;">
            <img src="data:image/png;base64,{logo_base64}" width="{width}" height="{height}">
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error(f"Logo file '{logo_path}' not found. Please ensure the logo file is in the same directory as the app.")

# read all pdf files and return text


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# split text into chunks


def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    return chunks  # list of strings

# get embeddings for each chunk


def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")  # type: ignore
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash",
                                   client=genai,
                                   temperature=0.3,
                                   )
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain


def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "upload some pdfs and ask me a question"}]


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")  # type: ignore

    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True) 
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True, )

    print(response)
    return response


def main():
    st.set_page_config(
        page_title="VariPhi RAG Engine",
        page_icon="🤖"
    )
    
    # Global CSS for black sidebar
    st.markdown("""
    <style>
    .stSidebar {
        background-color: #000000 !important;
    }
    .stSidebar > div {
        background-color: #000000 !important;
    }
    .stSidebar .stSelectbox > div > div {
        background-color: #000000 !important;
        color: white !important;
    }
    .stSidebar .stTextInput > div > div > input {
        background-color: #000000 !important;
        color: white !important;
    }
    .stSidebar .stButton > button {
        background-color: #333333 !important;
        color: white !important;
        border: 1px solid #555555 !important;
    }
    .stSidebar .stButton > button:hover {
        background-color: #444444 !important;
    }
    .stSidebar .stFileUploader > div {
        background-color: #000000 !important;
    }
    .stSidebar h1, .stSidebar h2, .stSidebar h3, .stSidebar h4, .stSidebar h5, .stSidebar h6 {
        color: white !important;
    }
    .stSidebar p, .stSidebar div {
        color: white !important;
    }
    .stSidebar .stSuccess {
        background-color: #1f4e3d !important;
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Sidebar for uploading PDF files
    with st.sidebar:
        # Display centered logo in sidebar
        logo_base64 = get_logo_base64("download-removebg-preview.png")
        if logo_base64:
            st.markdown(f"""
            <div style="text-align: center; margin-bottom: 20px; padding: 10px;">
                <img src="data:image/png;base64,{logo_base64}" width="60" height="60">
            </div>
            """, unsafe_allow_html=True)
        
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

    # Main content area for displaying chat messages
    st.title("VariPhi RAG Engine")
    st.write("Welcome to the chat!")
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    # Chat input
    # Placeholder for chat messages

    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "upload some pdfs and ask me a question"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Display chat messages and bot response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = user_input(prompt)
                placeholder = st.empty()
                full_response = ''
                for item in response['output_text']:
                    full_response += item
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)
        if response is not None:
            message = {"role": "assistant", "content": full_response}
            st.session_state.messages.append(message)


if __name__ == "__main__":
    main()
