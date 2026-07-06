import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. إعدادات الصفحة ودعم الواجهة العربية (RTL)
st.set_page_config(page_title="مساعد الملفات الذكي", page_icon="🤖", layout="centered")

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

st.title("🤖 مساعد المستندات الذكي (RAG App)")
st.subheader("ارفع ملفاتك واسألها مباشرة عبر السحابة")

# 2. التحقق من وجود مفتاح الـ API سحابياً لحماية التطبيق
if "OPENAI_API_KEY" not in os.environ:
    st.error("⚠️ لم يتم العثور على مفتاح 'OPENAI_API_KEY'. يرجى إضافته في إعدادات (Secrets) الـ Space الخاص بك.")
    st.stop()

# 3. تهيئة الذاكرة السحابية المؤقتة للمحادثة وقاعدة المتجهات
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# 4. واجهة رفع ومعالجة الملفات سحابياً
uploaded_file = st.file_uploader("اختر ملف PDF لبدء المعالجة", type=["pdf"])

if uploaded_file and st.session_state.vector_store is None:
    with st.spinner("جاري معالجة وتحليل الملف سحابياً..."):
        try:
            # حفظ مؤقت للملف داخل الحاوية السحابية لقراءته
            temp_file_path = "temp_uploaded_file.pdf"
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # استخراج النصوص وتقسيمها
            loader = PyPDFLoader(temp_file_path)
            docs = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            final_documents = text_splitter.split_documents(docs)
            
            # إنشاء الـ Embeddings وقاعدة بيانات المتجهات في الذاكرة السحابية
            embeddings = OpenAIEmbeddings()
            st.session_state.vector_store = FAISS.from_documents(final_documents, embeddings)
            
            # إزالة الملف المؤقت فوراً للأمان وحفظ المساحة
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                
            st.success("✅ تم تحليل المستند وبناء قاعدة بيانات المتجهات بنجاح!")
        except Exception as e:
            st.error(f"❌ حدث خطأ أثناء معالجة الملف: {str(e)}")

# 5. نظام الاستعلام والـ RAG
if st.session_state.vector_store is not None:
    # عرض سجل المحادثة المتوفر في الذاكرة
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # صندوق الاستقبال الذكي للمحادثة
    user_query = st.chat_input("اسأل أي شيء حول المستند...")
    
    if user_query:
        with st.chat_message("user"):
            st.write(user_query)
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        
        with st.spinner("جاري استخراج الإجابة الدقيقة..."):
            try:
                llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
                retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
                
                # توجيه النموذج للالتزام بالسياق المرفق فقط وباللغة العربية
                system_prompt = (
                    "أنت مساعد ذكي متخصص في الإجابة على الأسئلة بناءً على المستندات المرفقة فقط.\n"
                    "استخدم السياق التالي بدقة للإجابة على سؤال المستخدم. إذا لم تكن الإجابة موجودة في السياق، "
                    "أخبر المستخدم بوضوح ولطف أنك لا تملك الإجابة من خلال المستند المرفق، ولا تقم بابتكار إجابات.\n"
                    "يجب أن تكون الإجابة باللغة العربية، واضحة، ومباشرة.\n\n"
                    "{context}"
                )
                
                prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", "{input}"),
                ])
                
                question_answer_chain = create_stuff_documents_chain(llm, prompt)
                rag_chain = create_retrieval_chain(retriever, question_answer_chain)
                
                # تنفيذ الاستدعاء السحابي
                response = rag_chain.invoke({"input": user_query})
                answer = response["answer"]
                
                with st.chat_message("assistant"):
                    st.write(answer)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"⚠️ عذراً، حدث خطأ أثناء معالجة الطلب: {str(e)}")
else:
    if not uploaded_file:
        st.info("💡 يرجى رفع ملف PDF من الأعلى لتتمكن من البدء في طرح الأسئلة.")
