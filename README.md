# 🎥 AskChat — YouTube Video Q&A Chatbot

AskChat is an AI-powered chatbot built with **Streamlit**, **LangChain**, **FAISS**, and **Hugging Face LLMs** that allows you to ask questions about the content of any YouTube video using just the **video ID**.

It fetches the transcript of the video, semantically searches for relevant context, and generates intelligent answers using powerful language models.

![App Screenshot](assets\Screenshot 2025-07-20 170848.png)

---

## 🚀 Features

- 🔍 Ask any question about a YouTube video.
- 🧠 Powered by `meta-llama/Llama-3.1-8B-Instruct` via Hugging Face.
- 🌍 Multi-language transcript support (English, Hindi, Spanish, French, German).
- 💬 Natural language responses.
- 📚 Vector-based semantic search using FAISS.
- 🧩 Smart context filtering using LangChain.
- 🧼 Simple Streamlit interface.

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **Streamlit** – Web interface
- **LangChain** – Prompt + retriever chain
- **FAISS** – Vector database for semantic search
- **Hugging Face** – LLM inference (via Inference API)
- **YouTube Transcript API** – To fetch subtitles
- **Dotenv** – For secure token handling

---

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/askchat.git
   cd askchat
2. **Create a virtual environment:**
   conda create -n askchat python=3.13.4
   conda activate askchat
3. **Install dependencies:**
   pip install -r requirements.txt
4. **Create .env file and add your Hugging Face token:**
   HUGGINGFACEHUB_ACCESS_TOKEN=your_huggingface_api_token

---

## ▶️ Run the App
   streamlit run main.py

---

## Then open your browser at http://localhost:8501

---
