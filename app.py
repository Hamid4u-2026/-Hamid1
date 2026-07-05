# =============================================================================
# app.py
# =============================================================================
# مشروع بسيط تعليمي:
# PDF RAG Chat
#
# الفكرة:
# - رفع ملفات PDF
# - استخراج النص منها
# - تقسيم النص إلى أجزاء صغيرة
# - تحويلها إلى Embeddings
# - تخزينها في FAISS
# - عند السؤال: البحث عن أقرب أجزاء + إرسالها إلى Gemini
# =============================================================================

import os
import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# =============================================================================
# إعداد مفتاح Google Gemini API
# =============================================================================
# نحاول أولاً استخدام Secrets في Hugging Face
# وإذا لم يوجد نسمح للمستخدم بإدخاله يدوياً

GOOGLE_API_KEY = ""

try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except:
    pass

if not GOOGLE_API_KEY:
    st.sidebar.warning("⚠️ أدخل Google API Key")

api_input = st.sidebar.text_input("🔑 Google API Key", type="password")

if api_input:
    GOOGLE_API_KEY = api_input
    genai.configure(api_key=GOOGLE_API_KEY)

MODEL_NAME = "gemini-1.5-flash"

# =============================================================================
# تحميل نموذج Embedding (مرة واحدة فقط لتحسين الأداء)
# =============================================================================
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

# =============================================================================
# قراءة ملفات PDF واستخراج النص
# =============================================================================
def get_pdf_text(pdf_files):
    text = ""
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

# =============================================================================
# تقسيم النص إلى أجزاء صغيرة (Chunks)
# =============================================================================
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_text(text)

# =============================================================================
# إنشاء قاعدة بيانات FAISS
# =============================================================================
def create_vector_store(chunks):
    embeddings = load_embeddings()
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

# =============================================================================
# تحميل قاعدة البيانات
# =============================================================================
def load_vector_store():
    embeddings = load_embeddings()

    if not os.path.exists("faiss_index"):
        return None

    return FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

# =============================================================================
# البحث عن أفضل النصوص المشابهة
# =============================================================================
def search_docs(query):
    db = load_vector_store()

    if db is None:
        return []

    return db.similarity_search(query, k=4)

# =============================================================================
# إرسال السؤال إلى Gemini
# =============================================================================
def ask_gemini(context, question):

    if not GOOGLE_API_KEY:
        return "❌ يرجى إدخال مفتاح Gemini أولاً"

    prompt = f"""
أجب فقط من السياق التالي:

{context}

السؤال:
{question}

إذا لم تجد الإجابة قل: لا توجد إجابة في المستند
"""

    model = genai.GenerativeModel(MODEL_NAME)
    response = model.generate_content(prompt)

    return response.text

# =============================================================================
# واجهة Streamlit
# =============================================================================
st.set_page_config(page_title="PDF RAG Chat", layout="wide")

st.title("📄 PDF RAG Chat (تعليمي)")

user_question = st.text_input("💬 اكتب سؤالك هنا")

if user_question:

    docs = search_docs(user_question)

    context = "\n".join([d.page_content for d in docs])

    answer = ask_gemini(context, user_question)

    st.subheader("🤖 الإجابة")
    st.write(answer)

# =============================================================================
# Sidebar: رفع الملفات
# =============================================================================
st.sidebar.title("📁 رفع ملفات PDF")

pdf_files = st.sidebar.file_uploader(
    "ارفع ملفات PDF",
    type=["pdf"],
    accept_multiple_files=True
)

if st.sidebar.button("🚀 معالجة الملفات"):

    if pdf_files:

        with st.spinner("جاري المعالجة..."):

            raw_text = get_pdf_text(pdf_files)
            chunks = split_text(raw_text)
            create_vector_store(chunks)

        st.success("✅ تم تجهيز الملفات بنجاح!")

    else:
        st.warning("⚠️ يرجى رفع ملفات PDF أولاً")
