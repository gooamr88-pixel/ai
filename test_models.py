import google.generativeai as genai
import os
from dotenv import load_dotenv

# بنحمل المتغيرات من ملف .env
load_dotenv()

# لو المفتاح مش بيتقري، ممكن تمسح السطر اللي تحت وتحط المفتاح بين علامتين تنصيص مباشرة للتجربة
# api_key = "AIzaSyYourAPIKeyHere..."
api_key = os.getenv("AIzaSyAymKZf0T6oMO-ncIWrbk3VtthPM5Y2SPk") 

genai.configure(api_key=api_key)

print("🚀 جاري البحث عن الموديلات المتاحة لمفتاحك...\n")

try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"✅ الموديل: {m.name}")
except Exception as e:
    print(f"❌ حصل إيرور: {e}")