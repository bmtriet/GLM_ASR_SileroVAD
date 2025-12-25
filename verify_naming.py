
import os
import shutil
from pathlib import Path
from datetime import datetime
import torch
from asr_handler import ASRHandler
from segmentation import Segmenter

def test_new_naming():
    print("Testing new naming convention and folder structure...")
    
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
    folder_name = f"{base_name}_{timestamp}"
    custom_dir = Path("temp_uploads/segments") / folder_name
    
    print(f"Generated folder name: {folder_name}")
    
    # Run segmentation
    segment_files = segmenter.segment_to_files(audio_path, custom_dir=custom_dir)
    
    print(f"Generated {len(segment_files)} segments.")
    for seg in segment_files:
        print(f"File: {seg['filename']} | Start: {seg['start']}s | End: {seg['end']}s")
        # Verify filename format (simple check)
        assert "_" in seg['filename'], f"Filename missing separator: {seg['filename']}"
        assert seg['filename'].endswith(".wav"), f"Filename wrong extension: {seg['filename']}"

    # Verify directory content
    assert custom_dir.exists(), "Custom directory was not created."
    files_on_disk = list(custom_dir.glob("*.wav"))
    assert len(files_on_disk) == len(segment_files), f"Expected {len(segment_files)} files, found {len(files_on_disk)}"

    print("\nSUCCESS: Folder and filename conventions verified.")
    
    # Cleanup (Optional: uncomment to keep for manual check)
    # shutil.rmtree(custom_dir)

if __name__ == "__main__":
    test_new_naming()
