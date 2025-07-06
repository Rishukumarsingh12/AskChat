from langchain_community.llms import Ollama
from langchain_community.llms import HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint
from huggingface_hub import InferenceClient

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings


from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from langchain_core.prompts import PromptTemplate

from transformers import pipeline

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
import streamlit as st
import logging

client = InferenceClient(token="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")

st.set_page_config(page_title="AskTube",layout="centered")
st.title("YouTube video Q&A Chatbot")
st.markdown("Ask any questions about your given you tube video.")

video_id = st.text_input("Enter YouTube Video ID",placeholder="e.g. dQe4w9WgXcq")
query = st.text_input("Give me your query",placeholder="e.g. what is this video about")

# Basic configuration
logging.basicConfig(
    level=logging.INFO,  # Levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='app.log',  # Log to a file, or remove for console output
    filemode='w'         # 'w' = overwrite, 'a' = append
)


#video_id = "Gfr50f6ZBvo" # only the ID, not full URL
try:
    # If you don’t care which language, this returns the “best” one
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])

    # Flatten it to plain text
    transcript = " ".join(chunk["text"] for chunk in transcript_list)
    logging.info(f"Transcript : {transcript}")
    logging.info("Transcript is successfully fetched by api.")

except TranscriptsDisabled:
    #print("No captions available for this video.")
    logging.error("No captions available for this video.")

#TextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size = 512, chunk_overlap = 100)
chunks = splitter.create_documents([transcript])
logging.info("Chunks has been successfully created.")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectore_store = FAISS.from_documents(chunks, embeddings)
logging.info("chunks has been successfully stored in vector database.")

logging.info(f"{vectore_store.index_to_docstore_id}")
#logging.info(f"{vectore_store.get_by_ids(['3f507d39-ae93-4fac-88bc-2d547f7642fa'])}")
logging.info(f"length of chunks: {len(chunks)}")


retriever = vectore_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
retriever
#logging.info(f"{retriever.invoke('explain deepmind')}")
logging.info("Retriever is working good.")


#Augmentation
prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)

def augmentation_pipeline(question):
  retrieved_docs    = retriever.invoke(question)
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  final_prompt = prompt.invoke({"context": context_text, "question": question})
  return final_prompt

#question = "is the topic of nuclear fusion discussed in this video? if yes then what was discussed"
#final_prompt = augmentation_pipeline(question)
#logging.info(f"{final_prompt}")

"""
pipe = pipeline("text2text-generation", model="google/flan-t5-small", device=-1) 
llm = HuggingFacePipeline(pipeline=pipe)

llm = HuggingFaceEndpoint(
   repo_id="mistralai/Mistral-7B-Instruct-v0.1",
   task="text2text-generation",
   client = client
)
"""
llm = Ollama(model= "mistral")
#ans = llm.invoke(final_prompt)
#logging.info(f"{ans}")


def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text

parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})
#logging.info(f"{parallel_chain.invoke('who is Demis')}")
#logging.info("Parallel chains are working.")

parser = StrOutputParser()
main_chain = parallel_chain | prompt | llm | parser
#logging.info(f"{main_chain.invoke('Can you summarize the video')}")

if st.button("Submit query"):
   if not video_id or not query:
      st.warning("Please enter you tube video ID and query")
   else:
      st.spinner("Thinking")
      res = main_chain.invoke(query)
      st.markdown("Answer: ")
      st.write(res)