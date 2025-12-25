import requests
import os

URL = "https://localhost:8443/transcribe"
AUDIO_FILE = "test_audio.wav" # Ensure this exists or use a small wav file
CERT_PATH = "cert.pem" # If testing with SSL

def test_transcribe_no_segmentation():
    print("Testing /transcribe with segmentation=False...")
    with open(AUDIO_FILE, "rb") as f:
        files = {"file": f}
        data = {"segmentation": "false"}
        response = requests.post(URL, files=files, data=data, verify=False)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")

def test_transcribe_with_segmentation():
    print("\nTesting /transcribe with segmentation=True...")
    with open(AUDIO_FILE, "rb") as f:
        files = {"file": f}
        data = {"segmentation": "true"}
        response = requests.post(URL, files=files, data=data, verify=False)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")

if __name__ == "__main__":
    if not os.path.exists(AUDIO_FILE):
        print(f"Error: {AUDIO_FILE} not found. Please provide a test audio file.")
    else:
        test_transcribe_no_segmentation()
        test_transcribe_with_segmentation()
