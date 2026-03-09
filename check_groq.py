import os
from groq import Groq

API_KEY = ""

try:
    print("🔌 جاري الاتصال بـ Groq...")
    client = Groq(api_key=API_KEY)

    models = client.models.list()
    
    print("\n✅ تم الاتصال بنجاح! دي الموديلات المتاحة :")
    print("-" * 40)
    
    available_models = []
    for model in models.data:
        print(f"🌟 {model.id}")
        available_models.append(model.id)

    print("-" * 40)
    
    if available_models:
     
        test_model = available_models[0]
        print(f"\n🧪 جاري عمل اختبار سرعة باستخدام: {test_model}...")
        
        completion = client.chat.completions.create(
            model=test_model,
            messages=[
                {"role": "user", "content": "Say 'Hello from Groq Fast!'"}
            ]
        )
        print(f"🚀 الرد وصل: {completion.choices[0].message.content}")
        
    else:
        print("❌ غريبة! المفتاح شغال بس مفيش موديلات متاحة!")

except Exception as e:
    print(f"\n💣 خطأ في الاتصال: {e}")
    print("💡 تأكد إن المفتاح صح، وإن النت شغال.")