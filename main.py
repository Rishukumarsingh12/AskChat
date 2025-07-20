# app.py
import os
from custom_logger import CustomLogger
import streamlit as st
from dotenv import load_dotenv

from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled

from huggingface_hub import InferenceClient
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda

# Load .env token
load_dotenv()
log = CustomLogger("main_py.log")
token = os.getenv("HUGGINGFACEHUB_TOKEN")

# Initialize Hugging Face Inference Client
client = InferenceClient(
    model="meta-llama/Llama-3.1-8B-Instruct", # or novita, gradientai, etc.
    
)


# Streamlit page
st.set_page_config(page_title="AskTube", layout="centered")
st.title("YouTube video Q&A Chatbot")
st.markdown("Ask any question about the content of a YouTube video.")

# User Inputs
video_id = st.text_input("Enter YouTube Video ID", placeholder="e.g. dQw4w9WgXcQ")
query = st.text_input("Give me your query", placeholder="e.g. what is this video about")

language = st.selectbox(
    "Select transcript language:",
    options=[("English", "en"), ("Hindi", "hi"), ("Spanish", "es"), ("French", "fr"), ("German", "de")],
    format_func=lambda x: x[0],
    index=0
)
selected_language_code = language[1]

# Transcript Retrieval
try:
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=[selected_language_code])
    transcript = " ".join(chunk["text"] for chunk in transcript_list)
    log.info("Transcript successfully fetched.")
except NoTranscriptFound:
    st.error("No transcript found for this video.")
    log.info("No transcriptions found for this video.")
    st.stop()
except TranscriptsDisabled:
    st.error("Transcripts are disabled for this video.")
    log.info("Transcriptions are disabled for this video.")
    st.stop()

# Split transcript into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100)
chunks = splitter.create_documents([transcript])
log.info(f"{len(chunks)} chunks created.")

# Embed and store in vector DB
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# Prompt Template
prompt = PromptTemplate(
    template="""
    You are a helpful assistant.
    Answer ONLY from the provided transcript context.
    If the context is insufficient, just say you don't know.

    {context}
    Question: {question}
    """,
    input_variables=['context', 'question']
)

# Format context
def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)

# Create chain
parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})

# Create the full prompt
def build_prompt(query):
    inputs = parallel_chain.invoke(query)
    return prompt.format(**inputs)

# Chat model call using Hugging Face streaming client
def chat_with_llm(full_prompt: str):
    stream = client.chat.completions.create(
         # or other supported model
        messages=[
            {"role": "user", "content": full_prompt}
        ],
        stream=True,
        temperature=0.7,
        top_p=0.7,
    )

    full_response = ""
    for chunk in stream:
        content = chunk.choices[0].delta.content or ""
        full_response += content
    return full_response

# Submit logic
if st.button("Submit query"):
    if not video_id or not query:
        st.warning("Please enter both the YouTube Video ID and your question.")
    else:
        with st.spinner("Thinking..."):
            full_prompt = build_prompt(query)
            log.info(f"Final Prompt: {full_prompt}")
            response = chat_with_llm(full_prompt)
            st.markdown("### Answer:")
            st.write(response)
