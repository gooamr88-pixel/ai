import sys
print("Checking imports...")
try:
    import fitz
    print("PyMuPDF: OK")
except ImportError as e:
    print(f"PyMuPDF Error: {e}")

try:
    import google.generativeai
    print("Gemini: OK")
except ImportError as e:
    print(f"Gemini Error: {e}")

try:
    from app.main import app
    print("App Import: OK")
except Exception as e:
    print(f"App Import Error: {e}")

print("Done.")
