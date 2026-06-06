"""
Ruya — API Test Suite (Refactored)
==================================
Tests all modular endpoints after the backend refactoring.
Run: python test_api.py (with backend running at http://localhost:8000)
"""

import os
import requests
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
    print("\n[1/5] Health Check...")
    try:
        r = requests.get(f"{BASE_URL}/", timeout=5)
        data = r.json()
        assert r.status_code == 200
        assert data["status"] == "operational"
        assert "video-8min" in data["modules"]
        assert "podcast-8min" in data["modules"]
        print(f"  ✓ Status: {data['status']}, Modules: {data['modules']}")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")


def test_question_bank():
    """Test POST /api/v1/text/generate-question-bank."""
    print("\n[2/5] Question Bank Generation...")
    try:
        create_dummy_pdf()
        with open("test.pdf", "rb") as f:
            r = requests.post(
                f"{BASE_URL}/api/v1/text/generate-question-bank",
                files={"file": ("test.pdf", f, "application/pdf")},
                timeout=60,
            )
        assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
        data = r.json()
        assert "questions" in data
        print(f"  ✓ Generated {len(data['questions'])} questions successfully")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")


def test_mindmap():
    """Test POST /api/v1/text/generate-mindmap."""
    print("\n[3/5] Mind Map Generation...")
    try:
        create_dummy_pdf()
        with open("test.pdf", "rb") as f:
            r = requests.post(
                f"{BASE_URL}/api/v1/text/generate-mindmap",
                files={"file": ("test.pdf", f, "application/pdf")},
                timeout=60,
            )
        assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
        data = r.json()
        assert "mindmap_image_url" in data
        print(f"  ✓ Mindmap Image URL: {data['mindmap_image_url']}")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")


def test_video():
    """Test POST /api/v1/media/video/generate."""
    print("\n[4/5] Video Generation...")
    try:
        create_dummy_pdf()
        with open("test.pdf", "rb") as f:
            r = requests.post(
                f"{BASE_URL}/api/v1/media/video/generate",
                files=[("files", ("test.pdf", f, "application/pdf"))],
                timeout=240,
            )
        assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
        data = r.json()
        assert "final_video_url" in data
        print(f"  ✓ Video URL: {data['final_video_url']}")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")


def test_podcast():
    """Test POST /api/v1/media/podcast/generate."""
    print("\n[5/5] Podcast Generation...")
    try:
        create_dummy_pdf()
        with open("test.pdf", "rb") as f:
            r = requests.post(
                f"{BASE_URL}/api/v1/media/podcast/generate",
                files=[("files", ("test.pdf", f, "application/pdf"))],
                timeout=240,
            )
        assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
        data = r.json()
        assert "final_audio_url" in data
        print(f"  ✓ Podcast Audio URL: {data['final_audio_url']}")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")


if __name__ == "__main__":
    print("=" * 50)
    print("Ruya API Test Suite — v5.0")
    print("=" * 50)

    test_health()
    test_question_bank()
    test_mindmap()
    test_video()
    test_podcast()

    # Cleanup local test file
    if os.path.exists("test.pdf"):
        os.remove("test.pdf")

    print("\n" + "=" * 50)
    print("All tests complete.")
    print("=" * 50)
