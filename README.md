# 📄 PDF RAG Chat

An efficient, lightweight, and fast AI-powered application that allows you to chat with your PDF documents locally or via cloud platforms. This project is specifically designed to run seamlessly on low-spec hardware by offloading the heavy computational workload to the cloud using **Google AI Studio** and **GitHub Codespaces**.

---

## 🚀 Features
- **Smart Document Q&A:** Ask questions about uploaded PDF files and get precise answers based *only* on the document context.
- **Fast Execution:** Powered by Google's lightning-fast `gemini-1.5-flash` model.
- **Cloud-Powered:** Runs completely in the cloud using GitHub Codespaces, requiring zero installation or heavy resources on your local computer.
- **Strict Context Boundary:** Ensures the AI answers strictly from your provided documents, avoiding hallucinations.

---

## 🛠️ Architecture (RAG Workflow)
1. **Text Extraction:** Reads text content from uploaded PDFs page by page using `PyPDF2`.
2. **Text Chunking:** Splits large text blocks into optimized chunks using LangChain's `RecursiveCharacterTextSplitter`.
3. **Vector Embeddings:** Converts text into numerical vectors using a lightweight HuggingFace embedding model (`all-MiniLM-L6-v2`).
4. **Vector Store:** Saves and indexes text chunks locally in a `FAISS` vector store.
5. **Similarity Search:** Queries the FAISS index to retrieve the top 4 most relevant text pieces matching the user's question.
6. **Generation:** Sends the context and question to Google Gemini API to generate the final response.

---

## 💻 Tech Stack
- **Framework:** Streamlit
- **AI/LLM Orchestration:** LangChain & Google Generative AI
- **Vector Database:** FAISS (Facebook AI Similarity Search)
- **Model:** `gemini-1.5-flash`

---

## 📥 How to Run on GitHub Codespaces

1. **Upload Files:** Ensure both `app.py` and `requirements.txt` are inside this repository.
2. **Launch Codespace:** Click the green **Code** button at the top right, select the **Codespaces** tab, and click **Create codespace on main**.
3. **Install Dependencies:** Open the terminal in your Codespace and run:
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the Application:** Start the Streamlit server by executing:
   ```bash
   streamlit run app.py
   ```
5. **Open App:** Click the **Open in Browser** pop-up button in the bottom right corner to access the web interface.

---

## 🔑 Setup API Key (Recommended)
To avoid entering your Google API Key manually every time you open the app:
1. Go to your GitHub repository **Settings** -> **Secrets and variables** -> **Actions**.
2. Click **New repository secret**.
3. Set Name to `GOOGLE_API_KEY`.
4. Set Value to your personal Gemini API key from [Google AI Studio](https://google.com).
5. Click **Add secret**.

---
*Developed as a university-level AI project for learning RAG architectures.*
