FROM python:3.10-slim

# ضبط إعدادات البيئة لمنع تجميد المخرجات
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# إنشاء المستخدم الخاص ببيئة Hugging Face وضبط المجلد
RUN useradd -m -u 1000 user
WORKDIR $HOME/app

# نسخ ملف المكتبات أولاً لاستغلال خاصية الكاش السحابي السريع
COPY --chown=user:user requirements.txt .

# التبديل للمستخدم وتثبيت المكتبات
USER user
RUN pip install --no-cache-dir --user -r requirements.txt

# نسخ باقي ملفات المشروع إلى الحاوية
COPY --chown=user:user . .

# المنفذ الخاص باستقبال طلبات الويب
EXPOSE 7860

# أمر التشغيل المباشر لتطبيق Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
