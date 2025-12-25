
import torch
import torchaudio
import os
import shutil
import uuid
from asr_handler import ASRHandler
from segmentation import Segmenter
from pathlib import Path

def verify_full_cycle(audio_path):
    print(f"Verifying full cycle transcription for: {audio_path}")
    
    # 1. Initialize models (real ones for simulation if possible, else mock)
    # To be safe and fast for logic verification, we mock the ASR part
    # but use the real Segmenter to check the flow.
    
    class MockHandler:
        device = "cpu"
        def transcribe(self, audio):
            # In real app, this would be the GLM model
            return f"Transcribed {os.path.basename(audio)}"

    checkpoint_dir = "."
    handler = MockHandler()
    segmenter = Segmenter(handler)
    
    # 2. Simulate /transcribe-full endpoint logic
    print("\n--- Starting Full-Cycle Simulation ---")
    
    # Segment to files
    segment_results = segmenter.segment_to_files(audio_path)
    print(f"Segmented into {len(segment_results)} files.")
    
    final_segments = []
    full_text_parts = []
    
    for seg in segment_results:
        text = handler.transcribe(seg['path'])
        
        final_segments.append({
            "id": seg['id'],
            "start": seg['start'],
            "end": seg['end'],
            "text": text
        })
        full_text_parts.append(text)
        
        # Cleanup segment file
        if os.path.exists(seg['path']):
            os.remove(seg['path'])
            
    result = {
        "segments": final_segments,
        "full_text": " ".join(full_text_parts)
    }
    
    print("\n--- Final Output Summary ---")
    print(f"Total Segments: {len(result['segments'])}")
    print(f"First 2 segments: {result['segments'][:2]}")
    print(f"Full text snippet: {result['full_text'][:100]}...")
    
    return len(result['segments'])

if __name__ == "__main__":
    audio_file = "/home/gmo-admin/Desktop/Python_playground/GLM_ASR/e96f4de9-d452-4da6-8c45-e10356aa179d.wav"
    if os.path.exists(audio_file):
        count = verify_full_cycle(audio_file)
        if count == 10:
            print("\nSUCCESS: 10 segments correctly identified and processed.")
        else:
            print(f"\nFAILURE: Expected 10 segments, got {count}.")
    else:
        print(f"Error: File not found at {audio_file}")
