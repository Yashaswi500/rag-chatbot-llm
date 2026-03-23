import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import pipeline
from pypdf import PdfReader

# Page config
st.set_page_config(page_title="RAG Chatbot", layout="wide")

st.title("🤖 Smart RAG Chatbot")

# ---------------- PDF Upload ---------------- #
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
    text = ""
    pdf_reader = PdfReader(uploaded_file)

    for page in pdf_reader.pages:
        text += page.extract_text()

    st.success("PDF uploaded successfully ✅")

    # ---------------- TEXT SPLITTING ---------------- #
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=100
    )
    docs = text_splitter.create_documents([text])

    # ---------------- EMBEDDINGS ---------------- #
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # ---------------- VECTOR DB ---------------- #
    db = FAISS.from_documents(docs, embeddings)

    # ---------------- RETRIEVER ---------------- #
    retriever = db.as_retriever(search_kwargs={"k": 6})

    # ---------------- LLM ---------------- #
    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_new_tokens=100
    )
    llm = HuggingFacePipeline(pipeline=pipe)

    # ---------------- PROMPT ---------------- #
    prompt_template = """
You are a smart assistant.

Answer ONLY based on the given context.

Rules:
- If asked about skills → only list skills
- If asked about experience → only describe experience
- If asked for summary → give short summary
- If asked about contact → extract email, phone, location
- If answer not found → say "Not found in resume"
- Keep answer short and relevant

Context:
{context}

Question:
{question}

Answer:
"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # ---------------- QA CHAIN ---------------- #
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT}
    )

    # ---------------- CHAT HISTORY ---------------- #
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Show previous messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    user_input = st.chat_input("Ask me anything...")

    if user_input:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Get answer
        result = qa({"query": user_input})
        answer = result["result"]

        # Show bot response
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)

else:
    st.warning("Please upload a PDF to start chatting 📄")