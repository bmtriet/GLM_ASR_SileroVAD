import torch
import torchaudio
import gc
from pathlib import Path
from typing import List, Dict, Union, Any
from asr_handler import ASRHandler

class Segmenter:
    def __init__(self, asr_handler: ASRHandler, device: str = None):
        self.asr_handler = asr_handler
        self.device = device or asr_handler.device
        
        print("Loading Silero VAD model...")
        self.vad_model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True
        )
        (self.get_speech_timestamps, _, className, _, _) = utils
        self.vad_model.to(self.device)
        print("Silero VAD loaded.")

    def segment_to_files(self, audio_path: Union[str, Path], 
                         min_segment_sec: float = 1.0, 
                         max_segment_sec: float = 60.0,
                         min_speech_duration_ms: int = 250,
                         min_silence_duration_ms: int = 700,
                         custom_dir: Path = None) -> List[Dict[str, Any]]:
        """
        Segment audio using VAD on CPU and save to physical files.
        Returns a list of segment metadata (IDs, start, end, filename).
        """
        import os
        import uuid
        import gc
        
        # Ensure segments directory exists
        if custom_dir:
            segments_dir = custom_dir
        else:
            segments_dir = Path("temp_uploads/segments")
        
        segments_dir.mkdir(parents=True, exist_ok=True)

        # 1. Load audio metadata and full audio for VAD (CPU)
        print(f"Loading audio for segmentation: {audio_path}")
        wav, orig_sr = torchaudio.load(str(audio_path))
        
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
            
        if orig_sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=16000)
            wav_16k = resampler(wav)
        else:
            wav_16k = wav

        wav_16k = wav_16k.squeeze(0).cpu()

        # 2. Get speech timestamps (Running on CPU)
        print(f"Running VAD on CPU...")
        self.vad_model.to('cpu')
        with torch.no_grad():
            speech_timestamps = self.get_speech_timestamps(
                wav_16k, 
                self.vad_model, 
                sampling_rate=16000,
                min_speech_duration_ms=min_speech_duration_ms,
                min_silence_duration_ms=min_silence_duration_ms 
            )
        
        # Immediate cleanup of 16k buffer
        del wav_16k
        gc.collect()
        
        # 3. Merge segments
        merged_segments = self._merge_segments(speech_timestamps, 16000, min_segment_sec, max_segment_sec)
        
        # 4. Save segments and create a merged silence-optimized file
        results = []
        all_chunks = []
        current_merged_time = 0.0
        print(f"Total segments after splitting/merging: {len(merged_segments)}")
        
        for i, seg in enumerate(merged_segments):
            start_sample_16k = seg['start']
            end_sample_16k = seg['end']
            
            start_sec = start_sample_16k / 16000
            end_sec = end_sample_16k / 16000
            duration = end_sec - start_sec
            
            frame_offset = int(start_sec * orig_sr)
            num_frames = int(duration * orig_sr)
            
            # Avoid empty chunks
            if num_frames <= 0:
                continue
                
            chunk = wav[:, frame_offset : frame_offset + num_frames]
            all_chunks.append(chunk)
            
            # Use timestamp for filename: HH-MM-SS_HH-MM-SS.wav
            start_ts = self._format_timestamp(start_sec)
            end_ts = self._format_timestamp(end_sec)
            filename = f"{start_ts}_{end_ts}.wav"
            segment_path = segments_dir / filename
            
            # If filename exists
            if segment_path.exists():
                segment_path = segments_dir / f"{start_ts}_{end_ts}_{uuid.uuid4().hex[:4]}.wav"
                filename = segment_path.name

            torchaudio.save(str(segment_path), chunk, orig_sr)
            
            results.append({
                "id": str(uuid.uuid4()),
                "start": round(start_sec, 2),
                "end": round(end_sec, 2),
                "merged_start": round(current_merged_time, 2),
                "merged_end": round(current_merged_time + duration, 2),
                "filename": filename,
                "path": str(segment_path)
            })
            
            current_merged_time += duration
            
            # Clear individual chunk tensor from local scope (it's still in all_chunks)
            del chunk
            
        # Create and save merged file if speech detected
        merged_file_path = None
        if all_chunks:
            merged_wav = torch.cat(all_chunks, dim=1)
            merged_file_path = segments_dir / "merged.wav"
            torchaudio.save(str(merged_file_path), merged_wav, orig_sr)
            print(f"Merged audio saved to: {merged_file_path}")
            del merged_wav
            all_chunks.clear()

        # Cleanup
        del wav
        gc.collect()
        
        # Add merged file info to response
        final_response = {
            "segments": results,
            "merged_path": str(merged_file_path) if merged_file_path else None
        }
        
        return final_response

    def transcribe_with_segments(self, audio_path: Union[str, Path], 
                                 min_segment_sec: float = 1.0, 
                                 max_segment_sec: float = 60.0,
                                 min_speech_duration_ms: int = 250,
                                 min_silence_duration_ms: int = 700) -> List[Dict[str, Any]]:
        """
        Segment audio using VAD on CPU (safely), physically split into files, and transcribe.
        """
        import os
        import uuid
        import gc
        
        # Ensure segments directory exists
        segments_dir = Path("temp_uploads/segments")
        segments_dir.mkdir(parents=True, exist_ok=True)

        # 1. Load audio metadata and full audio for VAD (CPU)
        print(f"Loading audio from disk: {audio_path}")
        wav, orig_sr = torchaudio.load(str(audio_path))
        total_frames = wav.shape[1]
        
        print(f"Loaded: {audio_path} | SR: {orig_sr} | Total Frames: {total_frames}")

        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
            
        if orig_sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=16000)
            wav_16k = resampler(wav)
        else:
            wav_16k = wav

        # Ensure VAD input is on CPU and explicitly managed
        wav_16k = wav_16k.cpu()

        # 2. Get speech timestamps (Running on CPU for safety)
        print(f"Running VAD on CPU...")
        self.vad_model.to('cpu')
        
        with torch.no_grad():
            speech_timestamps = self.get_speech_timestamps(
                wav_16k, 
                self.vad_model, 
                sampling_rate=16000,
                min_speech_duration_ms=min_speech_duration_ms,
                min_silence_duration_ms=min_silence_duration_ms 
            )
        
        print(f"Raw VAD timestamps detected: {len(speech_timestamps)} segments")
        
        # Immediate cleanup of 16k buffer to save RAM
        del wav_16k
        gc.collect()
        
        # 3. Merge segments
        merged_segments = self._merge_segments(speech_timestamps, 16000, min_segment_sec, max_segment_sec)
        
        # 4. Transcribe each segment using physical file splitting
        results = []
        print(f"Processing {len(merged_segments)} segments using physical file splitting...")
        
        for i, seg in enumerate(merged_segments):
            start_sample_16k = seg['start']
            end_sample_16k = seg['end']
            
            # Convert 16k timestamps back to original sample rate for accurate splitting
            start_sec = start_sample_16k / 16000
            end_sec = end_sample_16k / 16000
            
            frame_offset = int(start_sec * orig_sr)
            num_frames = int((end_sec - start_sec) * orig_sr)
            
            # Extract chunk from original wav (still in CPU memory)
            chunk = wav[:, frame_offset : frame_offset + num_frames]
            
            # Save chunk to physical file
            segment_id = str(uuid.uuid4())
            segment_path = segments_dir / f"{segment_id}.wav"
            torchaudio.save(str(segment_path), chunk, orig_sr)
            
            # Transcribe from physical file
            # GLM ASR handler in this repo supports str/Path or Tensor.
            text = self.asr_handler.transcribe(str(segment_path))
            
            segment_result = {
                "id": i,
                "start": round(start_sec, 2),
                "end": round(end_sec, 2),
                "text": text
            }
            results.append(segment_result)
            print(f"Segment {i}: {segment_result['start']}-{segment_result['end']}s | {text}")
            
            # Cleanup physical file immediately
            if segment_path.exists():
                os.remove(segment_path)
            
            # Explicit cleanup after each segment
            del chunk
            torch.cuda.empty_cache()
            
        # Final cleanup of the full audio tensor
        del wav
        gc.collect()
        
        return results

    def _merge_segments(self, timestamps, sr, min_sec, max_sec):
        """
        Robustly split and merge VAD segments.
        Prioritizes precise VAD boundaries.
        Only splits if a segment exceeds max_sec.
        Only merges if segments are extremely close (optional/minimal).
        """
        if not timestamps:
            return []
            
        max_samples = int(max_sec * sr)
        
        # Pass 1: Handle long segments by splitting them
        # We also keep these as separate entries to respect the original VAD boundaries
        processed_segments = []
        for ts in timestamps:
            start = ts['start']
            end = ts['end']
            
            # If a single VAD segment is longer than max_samples, split it
            while (end - start) > max_samples:
                processed_segments.append({'start': start, 'end': start + max_samples})
                start += max_samples
            
            if start < end:
                processed_segments.append({'start': start, 'end': end})
        
        # Pass 2: Optional merging of extremely close segments (e.g., < 200ms)
        # For now, let's keep it SIMPLE: we don't merge across silence unless forced.
        # This fixes the "10 numbers" case where gaps are meaningful.
        
        return processed_segments

    def _format_timestamp(self, seconds: float) -> str:
        """Helper to format seconds into HH-MM-SS."""
        td = int(seconds)
        hh = td // 3600
        mm = (td % 3600) // 60
        ss = td % 60
        ms = int((seconds - int(seconds)) * 100)
        return f"{hh:02d}-{mm:02d}-{ss:02d}.{ms:02d}"
