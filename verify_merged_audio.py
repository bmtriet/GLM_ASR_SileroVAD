
import os
import shutil
from pathlib import Path
from datetime import datetime
import torch
import torchaudio
from asr_handler import ASRHandler
from segmentation import Segmenter

def test_merged_audio():
    print("Testing audio merging and concatenated output...")
    
    # Setup
    audio_path = "/home/gmo-admin/Desktop/Python_playground/GLM_ASR/e96f4de9-d452-4da6-8c45-e10356aa179d.wav"
    if not os.path.exists(audio_path):
        print(f"Error: {audio_path} not found.")
        return

    class MockHandler:
        device = "cpu"
        def transcribe(self, audio):
            return "Mock"

    handler = MockHandler()
    segmenter = Segmenter(handler)
    
    # Simulate folder generation
    base_name = Path(audio_path).stem
    timestamp = datetime.now().strftime("%y%m%d%H%M%S")
    folder_name = f"{base_name}_merged_test_{timestamp}"
    custom_dir = Path("temp_uploads/segments") / folder_name
    custom_dir.mkdir(parents=True, exist_ok=True)
    
    # Run segmentation
    print(f"Running segmenter with custom_dir: {custom_dir}")
    result = segmenter.segment_to_files(audio_path, custom_dir=custom_dir)
    
    segments = result['segments']
    merged_path = result['merged_path']
    
    print(f"Generated {len(segments)} segments.")
    print(f"Merged path reported: {merged_path}")
    
    # Verify merged file exists
    assert merged_path is not None, "Merged path is missing from response."
    assert os.path.exists(merged_path), f"Merged file not found on disk: {merged_path}"
    
    # Verify merged file duration (should be sum of segments roughly)
    merged_info = torchaudio.info(merged_path)
    print(f"Merged audio duration: {merged_info.num_frames / merged_info.sample_rate:.2f}s")
    
    # Verify segments exist
    for seg in segments:
        assert os.path.exists(seg['path']), f"Segment file missing: {seg['path']}"

    print("\nSUCCESS: Audio merging verified successfully.")
    
    # Cleanup
    # shutil.rmtree(custom_dir)

if __name__ == "__main__":
    test_merged_audio()
