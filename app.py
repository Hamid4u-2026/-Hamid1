# =========================================================================
# 1. استيراد المكتبات الأدوات اللازمة للبرنامج
# =========================================================================
import os                          # للتعامل مع نظام التشغيل وقراءة الملفات والمجلدات
import streamlit as st             # المكتبة السحرية لبناء واجهات المستخدم الرسومية لبرامج الذكاء الاصطناعي بسهولة
from dotenv import load_dotenv     # لقراءة مفاتيح الـ API السرية من ملفات الإعدادات الخفية (.env)
from PyPDF2 import PdfReader       # مكتبة مخصصة لقراءة نصوص ملفات الـ PDF واستخراجها صفحة بصفحة

# استيراد أدوات LangChain لتسهيل التعامل مع النصوص وقواعد البيانات المتجهة (Vectors)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# استيراد مكتبة جوجل الرسمية للاتصال بنموذج Gemini
import google.generativeai as genai

# تفعيل خاصية تحميل المتغيرات السرية من النظام
load_dotenv()

# =========================================================================
# 2. إعداد الاتصال بنموذج Google Gemini السريع والمجاني
# =========================================================================
# قراءة مفتاح الـ API الخاص بجوجل من النظام (إذا تم حفظه في إعدادات GitHub Secrets)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if GOOGLE_API_KEY:
    # إذا وجدنا المفتاح، نقوم بربطه وتفعيله تلقائياً مع مكتبة جوجل
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    # إذا لم يجده، سنظهر تحذيراً للمستخدم في القائمة الجانبية لإدخاله يدوياً
    st.sidebar.warning("⚠️ يرجى إدخال مفتاح Google API Key في القائمة الجانبية للتشغيل.")

# تحديد النموذج الأسرع والأفضل والمناسب تماماً للتشغيل السحابي
MODEL_NAME = "gemini-1.5-flash"


# =========================================================================
# 3. الدوال البرمجية (Functions) لمعالجة الـ PDF والبحث والتوليد
# =========================================================================

def get_pdf_text(pdf_docs):
    """
    دالة تقوم بفتح ملفات الـ PDF المرفوعة واستخراج كافة النصوص بداخلها.
    """
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)       # قراءة الملف الحصري الحالي
        for page in pdf_reader.pages:     # المرور على كل صفحات الملف صفحة تلو الأخرى
            page_text = page.extract_text() # استخراج النص من الصفحة الحالية
            if page_text:
                text += page_text         # دمج النص المستخرج مع النصوص السابقة
    return text                           # إرجاع النص الكامل للكتاب أو الملف


def get_text_chunks(text):
    """
    دالة تقسم النص الضخم إلى قطع صغيرة؛ لأن حواسيب الذكاء الاصطناعي تفضل قراءة النصوص مجزأة بدقة.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,     # حجم كل قطعة نصية هو 1000 حرف تقريباً
        chunk_overlap=200    # تداخل بمقدار 200 حرف بين القطعة والأخرى لضمان عدم ضياع السياق في الأطراف
    )
    return splitter.split_text(text)


def get_vector_store(text_chunks):
    """
    دالة تحول النصوص إلى أرقام (Embeddings) يفهمها الكمبيوتر، ثم تخزنها في قاعدة بيانات محلية سريعة البحث.
    """
    # استخدام نموذج تشفير خفيف جداً ومجاني من HuggingFace ليعمل بسلاسة على السيرفر
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    # إنشاء قاعدة بيانات FAISS ووضع القطع النصية المشفرة بداخلها
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    # حفظ قاعدة البيانات في مجلد محلي باسم "faiss_index" لكي نعود إليها عند السؤال
    vector_store.save_local("faiss_index")


def ask_gemini(context, question):
    """
    دالة صياغة الأمر (Prompt Engineering) والاتصال بخوادم جوجل لطلب الإجابة الذكية السريعة.
    """
    if not GOOGLE_API_KEY:
        return "الرجاء إعداد مفتاح Google API أولاً لتتمكن من سؤال النموذج."

    # هندسة الأوامر: إعطاء تعليمات صارمة لـ Gemini لكي يجيب فقط من الملف المرفوع وبأمانة علمية
    prompt = f"""
You are a helpful AI assistant.

Answer ONLY using the provided context.

If the answer cannot be found in the context, reply exactly:
Answer is not available in the context.

Context:
{context}

Question:
{question}

Answer:
"""
    try:
        # استدعاء نموذج جينوم السحابي وإرسال النص والسؤال له
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(prompt)
        return response.text # إرجاع نص الإجابة القادم من جوجل
    except Exception as e:
        return f"حدث خطأ أثناء الاتصال بـ Google AI Studio: {str(e)}"


def user_input(user_question):
    """
    الدالة المحركة للـ (RAG): تبحث في قاعدة البيانات عن النصوص المناسبة لسؤالك ثم ترسلها للذكاء الاصطناعي لترجمتها لإجابة.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # حماية البرنامج: التأكد من أن المستخدم رفع ملفات وصنع قاعدة بيانات قبل أن يسأل
    if not os.path.exists("faiss_index"):
        st.error("❌ الرجاء رفع ملفات PDF ومعالجتها من القائمة الجانبية أولاً!")
        return

    # تحميل قاعدة البيانات المخزنة محلياً
    db = FAISS.load_local(
        "faiss_index", 
        embeddings, 
        allow_dangerous_deserialization=True # السماح بفتح الملفات المحلية بأمان
    )
    
    # البحث عن أفضل 4 قطع نصية داخل الـ PDF تشبه وتشرح سؤال المستخدم
    docs = db.similarity_search(user_question, k=4)
    # دمج هذه القطع الأربع معاً لتشكل "السياق" أو الغش المسموح به للنموذج
    context = "".join([doc.page_content for doc in docs])
    
    # إظهار دائرة تحميل متحركة للمستخدم أثناء معالجة خوادم جوجل للإجابة
    with st.spinner("🧠 جاري التفكير والبحث عبر خوادم Google السحابية..."):
        output = ask_gemini(context, user_question)
        # طباعة الإجابة النهائية بشكل منسق وجميل في الواجهة
        st.write("🤖 **الإجابة:**")
        st.info(output)

# =========================================================================
# 4. دالة تشغيل واجهة البرنامج الرسومية (Main Interface)
# =========================================================================
def main():
    # إعدادات الصفحة الأساسية في متصفح الويب
    st.set_page_config(page_title="PDF RAG Chat", layout="wide")
    # عنوان البرنامج المدمج والمختصر والمطلوب
    st.header("📄 PDF RAG Chat")

    # جعل المتغير العالمي للمفتاح متاحاً للتعديل داخل الواجهة
    global GOOGLE_API_KEY
    
    # مربع النص الأساسي لكتابة سؤال المستخدم
    user_question = st.text_input("💬 اكتب سؤالك حول المستندات المرفوعة هنا واضغط Enter:")
    if user_question:
        user_input(user_question) # استدعاء دالة البحث والإجابة فور كتابة السؤال

    # بناء القائمة الجانبية (Sidebar) لتنظيم العناصر والمستندات
    with st.sidebar:
        st.title("⚙️ لوحة التحكم")
        
        # حقل سري لإدخال مفتاح الـ API يدويًا إذا لم يقم الطالب بضبطه في الـ Secrets
        api_input = st.text_input("🔑 أدخل مفتاح Google API Key الخاص بك:", type="password")
        if api_input:
            GOOGLE_API_KEY = api_input
            genai.configure(api_key=GOOGLE_API_KEY)

        # أداة لرفع ملفات الـ PDF (تسمح برفع عدة ملفات معاً)
        pdf_docs = st.file_uploader(
            "📚 ارفع ملفات الـ PDF الخاصة بك:", 
            accept_multiple_files=True, 
            type=["pdf"]
        )
        
        # زر بدء معالجة الملفات المرفوعة
        if st.button("🚀 بدء معالجة الملفات"):
            if pdf_docs:
                with st.spinner("⏳ جاري قراءة وتقسيم وتشفير الملفات..."):
                    raw_text = get_pdf_text(pdf_docs)       # خطوة 1: استخراج النص
                    text_chunks = get_text_chunks(raw_text) # خطوة 2: تقسيم النص
                    get_vector_store(text_chunks)           # خطوة 3: التشفير والحفظ في FAISS
                    st.success("✅ تمت العملية بنجاح! يمكنك البدء بالأسئلة الآن.")
            else:
                st.warning("⚠️ من فضلك اختر ملف PDF واحد على الأقل أولاً.")

# نقطة انطلاق تشغيل الملف البرمجي بالكامل
if __name__ == "__main__":
    main()
