
import torch
import torchaudio
import os
from asr_handler import ASRHandler
from segmentation import Segmenter

def verify_segmentation(audio_path):
    print(f"Verifying segmentation for: {audio_path}")
    
    # Mock ASR Handler since we only care about segmentation count/timing
    class MockHandler:
        device = "cpu"
        def transcribe(self, audio):
            return "Mock"

    handler = MockHandler()
    segmenter = Segmenter(handler)
    
    # Run segmentation
    segment_files = segmenter.segment_to_files(audio_path)
    
    print(f"\nTotal segment files generated: {len(segment_files)}")
    for seg in segment_files:
        print(f"Segment {seg['id']}: {seg['start']}s - {seg['end']}s | File: {seg['filename']}")
    
    # Cleanup segment files
    for seg in segment_files:
        if os.path.exists(seg['path']):
            os.remove(seg['path'])
            
    return len(segment_files)

if __name__ == "__main__":
    audio_file = "/home/gmo-admin/Desktop/Python_playground/GLM_ASR/e96f4de9-d452-4da6-8c45-e10356aa179d.wav"
    if os.path.exists(audio_file):
        count = verify_segmentation(audio_file)
        print(f"\nFinal count for user audio: {count}")
    else:
        print(f"Error: File not found at {audio_file}")
