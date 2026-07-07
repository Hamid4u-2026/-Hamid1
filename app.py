import os
import streamlit as st
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. إعدادات الصفحة ودعم الواجهة العربية (RTL)
st.set_page_config(page_title="مساعد الملفات الذكي ", page_icon="🤖", layout="centered")

st.markdown("""
    <style>
    body, .main, .block-container, div[data-testid="stChatMessage"] {
        direction: RTL;
        text-align: right;
    }
    .stAlert {
        direction: RTL;
        text-align: right;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🤖 مساعد المستندات الذكي المجاني (RAG App)")
st.subheader("ارفع ملفاتك واسألها مجاناً بالكامل عبر السحابة")

# 2. التحقق من وجود مفتاح Hugging Face
if "HF_TOKEN" not in os.environ:
    st.error("⚠️ لم يتم العثور على مفتاح 'HF_TOKEN' في الإعدادات السحابية (Secrets) الخاصة بـ Space.")
    st.stop()

# 3. تهيئة الذاكرة السحابية المؤقتة للمحادثة وقاعدة المتجهات
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# 4. واجهة رفع ومعالجة الملفات سحابياً
uploaded_file = st.file_uploader("اختر ملف PDF لبدء المعالجة", type=["pdf"])

if uploaded_file and st.session_state.vector_store is None:
    with st.spinner("جاري معالجة وتحليل الملف سحابياً عبر النماذج المجانية..."):
        try:
            # استخدام مسار مؤقت آمن لتفادي Permission Denied
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.getbuffer())
                temp_file_path = temp_file.name
            
            loader = PyPDFLoader(temp_file_path)
            docs = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
            final_documents = text_splitter.split_documents(docs)
            
            # 💡 تحديث: تفعيل trust_remote_code لتخطي مشكلة النموذج الأمني لـ Alibaba
            embeddings = HuggingFaceEmbeddings(
                model_name="Alibaba-NLP/gte-multilingual-base",
                model_kwargs={"trust_remote_code": True}
            )
            st.session_state.vector_store = FAISS.from_documents(final_documents, embeddings)
            
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                
            st.success("✅ تم تحليل المستند وبناء قاعدة البيانات المجانية بنجاح!")
        except Exception as e:
            st.error(f"❌ حدث خطأ أثناء معالجة الملف: {str(e)}")

# 5. نظام الاستعلام والـ RAG المجاني
if st.session_state.vector_store is not None:
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    user_query = st.chat_input("اسأل أي شيء حول المستند...")
    
    if user_query:
        with st.chat_message("user"):
            st.write(user_query)
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        
        with st.spinner("جاري استخراج الإجابة الذكية مجاناً..."):
            try:
                # استدعاء نموذج محادثة (Qwen-2.5) مجاناً بالكامل
                llm = HuggingFaceHub(
                    repo_id="Qwen/Qwen2.5-7B-Instruct",
                    model_kwargs={"temperature": 0.1, "max_new_tokens": 512},
                    huggingfacehub_api_token=os.environ["HF_TOKEN"]
                )
                
                retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
                
                system_prompt = (
                    "<|im_start|>system\n"
                    "أنت مساعد ذكي متخصص في الإجابة على الأسئلة بناءً على المستندات المرفقة فقط.\n"
                    "استخدم السياق التالي بدقة للإجابة على سؤال المستخدم باللغة العربية. إذا لم تجد الإجابة، قل لا أعرف ولطفاً لا تبتكر.\n\n"
                    "السياق:\n{context}<|im_end|>\n"
                    "<|im_start|>user\n{input}<|im_end|>\n"
                    "<|im_start|>assistant\n"
                )
                
                prompt = ChatPromptTemplate.from_messages([
                    ("text", system_prompt)
                ])
                
                question_answer_chain = create_stuff_documents_chain(llm, prompt)
                rag_chain = create_retrieval_chain(retriever, question_answer_chain)
                
                response = rag_chain.invoke({"input": user_query})
                # تنظيف النص المرجوع ليعرض الإجابة فقط
                answer = response["answer"].split("<|im_start|>assistant\n")[-1].replace("<|im_end|>", "").strip()
                
                with st.chat_message("assistant"):
                    st.write(answer)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"⚠️ عذراً، حدث خطأ أثناء معالجة الطلب: {str(e)}")
else:
    if not uploaded_file:
        st.info("💡 يرجى رفع ملف PDF من الأعلى لتتمكن من البدء في طرح الأسئلة مجاناً.")
