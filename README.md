# 📄 PDF RAG Chat (مشروع تعليمي بسيط)

## 📌 فكرة المشروع

هذا مشروع بسيط يستخدم تقنية **RAG (Retrieval-Augmented Generation)** بحيث يمكنك:

- رفع ملفات PDF
- استخراج النصوص منها
- تقسيم النصوص إلى أجزاء صغيرة
- البحث داخل هذه الأجزاء
- استخدام Google Gemini للإجابة عن الأسئلة

---

## ⚙️ التقنيات المستخدمة

- Streamlit (واجهة المستخدم)
- FAISS (قاعدة بيانات متجهية)
- LangChain (تقسيم النصوص)
- Sentence Transformers (Embeddings)
- Google Gemini API (للإجابات الذكية)
- PyPDF2 (قراءة ملفات PDF)

---

## 🚀 كيفية تشغيل المشروع محليًا

### 1) تثبيت المكتبات

```bash
pip install -r requirements.txt
