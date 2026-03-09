"""
Ruya — API Test Suite
=======================
Tests all modular endpoints after the backend refactoring.
Run: python test_api.py (with backend running at http://localhost:8000)
"""

import requests
import json
import fitz  # PyMuPDF

BASE_URL = "http://localhost:8000"

SAMPLE_TEXT = (
    "Photosynthesis is the process used by plants, algae and certain bacteria "
    "to harness energy from sunlight and turn it into chemical energy. "
    "There are two types of photosynthetic processes: oxygenic photosynthesis "
    "and anoxygenic photosynthesis. The general equation for photosynthesis is: "
    "6CO2 + 6H2O + Light Energy -> C6H12O6 + 6O2. "
    "This process takes place in the chloroplasts, specifically using chlorophyll, "
    "the green pigment involved in photosynthesis."
)


def create_dummy_pdf(filename="test.pdf"):
    """Create a test PDF with sample educational text."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), SAMPLE_TEXT)
    doc.save(filename)
    print(f"  Created {filename}")


def test_health():
    """Test GET / health check."""
    print("\n[1/6] Health Check...")
    try:
        r = requests.get(f"{BASE_URL}/", timeout=5)
        data = r.json()
        assert r.status_code == 200
        assert data["status"] == "operational"
        assert "video" in data["modules"]
        assert "podcast" in data["modules"]
        print(f"  ✓ Status: {data['status']}, Modules: {data['modules']}")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")


def test_quiz():
    """Test POST /api/v1/text/quiz."""
    print("\n[2/6] Quiz Generation...")
    try:
        r = requests.post(
            f"{BASE_URL}/api/v1/text/quiz",
            json={"text": SAMPLE_TEXT, "num_questions": 3, "difficulty": "medium"},
            timeout=60,
        )
        assert r.status_code == 200
        data = r.json()
        assert "questions" in data
        print(f"  ✓ Generated {len(data['questions'])} questions")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")


def test_mindmap():
    """Test POST /api/v1/text/mindmap."""
    print("\n[3/6] Mind Map Generation...")
    try:
        r = requests.post(
            f"{BASE_URL}/api/v1/text/mindmap",
            json={"text": SAMPLE_TEXT},
            timeout=60,
        )
        assert r.status_code == 200
        data = r.json()
        assert "root_node" in data
        print(f"  ✓ Root node: {data['root_node'].get('label', 'N/A')}")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")


def test_upload():
    """Test POST /api/v1/text/upload with a dummy PDF."""
    print("\n[4/6] File Upload (PDF)...")
    try:
        create_dummy_pdf()
        with open("test.pdf", "rb") as f:
            r = requests.post(
                f"{BASE_URL}/api/v1/text/upload",
                files={"file": ("test.pdf", f, "application/pdf")},
                timeout=30,
            )
        assert r.status_code == 200
        data = r.json()
        assert data.get("success") is True
        print(f"  ✓ Extracted {len(data.get('text', ''))} chars from PDF")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")


def test_video():
    """Test POST /api/v1/media/video/generate."""
    print("\n[5/6] Video Generation...")
    try:
        r = requests.post(
            f"{BASE_URL}/api/v1/media/video/generate",
            json={"text": SAMPLE_TEXT, "num_segments": 3},
            timeout=120,
        )
        assert r.status_code == 200
        data = r.json()
        assert "segments" in data
        has_audio = any(s.get("audio_base64") for s in data["segments"])
        print(f"  ✓ Generated {len(data['segments'])} segments, audio={'yes' if has_audio else 'no'}")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")


def test_podcast():
    """Test POST /api/v1/media/podcast/generate."""
    print("\n[6/6] Podcast Generation...")
    try:
        r = requests.post(
            f"{BASE_URL}/api/v1/media/podcast/generate",
            json={"text": SAMPLE_TEXT, "num_turns": 4, "style": "educational"},
            timeout=120,
        )
        assert r.status_code == 200
        data = r.json()
        assert "turns" in data
        has_audio = any(t.get("audio_base64") for t in data["turns"])
        print(f"  ✓ Generated {len(data['turns'])} turns, audio={'yes' if has_audio else 'no'}")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")


if __name__ == "__main__":
    print("=" * 50)
    print("Ruya API Test Suite — v4.0")
    print("=" * 50)

    test_health()
    test_quiz()
    test_mindmap()
    test_upload()
    test_video()
    test_podcast()

    print("\n" + "=" * 50)
    print("All tests complete.")
    print("=" * 50)
