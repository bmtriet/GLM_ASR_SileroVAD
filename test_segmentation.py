
import torch
import torchaudio
import numpy as np
from asr_handler import ASRHandler
from segmentation import Segmenter
import os

def generate_test_audio(filename="test_audio.wav"):
    url = "https://models.silero.ai/vad_models/en.wav"
    print(f"Downloading example audio from {url}...")
    torch.hub.download_url_to_file(url, filename)
    return filename

def test_segmentation():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Generate audio
    audio_path = generate_test_audio()
    
    try:
        # Initialize
        handler = ASRHandler("model_path_placeholder", device=device) # User will have to adjust path or we rely on default handling if we can
        # Wait, the ASRHandler requires a valid checkpoint path or it tries to load from it. 
        # Since I don't have the weights here, the ASRHandler will probably fail to load the AutoModel.
        # However, for the purpose of checking the SEGMENTATION logic, I can mock the transcriber or expect it to fail if model missing.
        # But the user asked for "production-ready code".
        # I should assume the user has the model. 
        # But I need to run this to verify.
        # I will check if I can run it.
        pass
    except Exception as e:
        pass

    # Actually, proper test for ME to run:
    # I cannot load the ~GB model potentially.
    # So I will mock the ASR part for the test script verification if I can't find the model.
    # But for the USABLE script for the user, it should just work.
    
    # Getting real: The user's metadata shows "run.sh" running for 42m. This implies they might have the environment up.
    # But I don't know where the model is.
    # In 'app.py': checkpoint_dir = os.environ.get("CHECKPOINT_DIR", ".")
    # So it looks in current dir.
    
    # I will write the script to allow passing a model path or using "."
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default=".")
    args = parser.parse_args()
    
    audio_path = generate_test_audio()
    
    if not os.path.exists(os.path.join(args.model_dir, "config.json")) and not os.path.exists(os.path.join(args.model_dir, "model.safetensors")): 
         # Rough check if model exists. HuggingFace models usually have config.json.
         print("Warning: Model not found in current directory. Segmentation test will run but transcription might fail or need mocking.")
         # I'll create a MockHandler for demonstration if real one fails
         class MockHandler:
             device = "cpu"
             def transcribe(self, audio):
                 return "This is a simulated transcription of the speech segment."
         
         print("Using MOCK ASR Handler for testing logic...")
         handler = MockHandler()
    else:
        handler = ASRHandler(args.model_dir)

    segmenter = Segmenter(handler)
    
    # Test 1: Full pipeline (for compatibility)
    print("\nTesting full transcribe_with_segments pipeline...")
    segments = segmenter.transcribe_with_segments(audio_path)
    print(f"Total segments: {len(segments)}")
    
    # Test 2: Robust splitting in segment_to_files
    print("\nTesting segment_to_files with forced splitting (max 10s)...")
    # Setting max_segment_sec to 5.0 to force many splits even on a short test audio
    segment_files = segmenter.segment_to_files(audio_path, max_segment_sec=5.0)
    print(f"Total segment files generated: {len(segment_files)}")
    
    for seg in segment_files:
        duration = seg['end'] - seg['start']
        print(f"Segment {seg['id']}: {seg['start']}s - {seg['end']}s | Dur: {duration:.2f}s | File: {seg['filename']}")
        assert duration <= 5.1, f"Segment too long: {duration}s"
        
    assert len(segment_files) > len(segments), "Forced splitting should have produced more segments."
    print("\nTest Passed!")
    
    # Optional cleanup of test audio
    # os.remove(audio_path)
