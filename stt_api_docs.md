# GLM-ASR Speech-to-Text API Documentation

This document describes the Speech-to-Text (STT) API endpoints. For long files or precise segmentation, use the **Full-Cycle** endpoint.

## Base URL
Default: `https://0.0.0.0:8443`

> [!NOTE]
> The server uses self-signed SSL certificates. Disable SSL verification in your client for development.

---

## Primary Endpoint: Full-Cycle Transcription
**Endpoint**: `/transcribe-full`
**Method**: `POST`
**Content-Type**: `multipart/form-data`

This is the recommended endpoint for all use cases. It seamlessly handles segmentation (VAD) and transcription in a memory-safe sequence.

### Request Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `file` | `File` | **Required** | The audio file to transcribe. |
| `min_speech_ms` | `integer` | `250` | Min speech duration. |
| `min_silence_ms` | `integer` | `700` | Min silence to split segments. |
 
### Response
```json
{
  "folder": "temp_uploads/segments/audio_filename_231223150507",
  "merged_audio_url": "/outputs/segments/audio_filename_231223150507/merged.wav",
  "segments": [
    {
      "id": "uuid-1",
      "start": 0.0,
      "end": 2.5,
      "merged_start": 0.0,
      "merged_end": 2.5,
      "text": "Hello world"
    },
    ...
  ],
  "full_text": "Hello world. This is the complete transcription."
}
```

> [!TIP]
> **Segment Length**: By default, segments are split at silence gaps. If speech is continuous, segments will be capped at **60 seconds** for stability.

---

## Technical Endpoints (Decoupled)
If you need manual control over the process, you can use these separate endpoints.

## Step 1: Segment Audio
**Endpoint**: `/segment`
**Method**: `POST`
**Content-Type**: `multipart/form-data`

Splits a long audio file into small segments using Silero VAD on the CPU.

### Request Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `file` | `File` | **Required** | The long audio file. |
| `min_speech_ms` | `integer` | `250` | Min duration of speech (ms). |
| `min_silence_ms` | `integer` | `700` | Min duration of silence to split (ms). |

### Response
```json
{
  "folder": "temp_uploads/segments/audio_filename_231223150507",
  "segments": [
    {
      "id": "uuid-1",
      "start": 0.0,
      "end": 2.5,
      "filename": "00-00-00.00_00-00-02.50.wav",
      "path": "temp_uploads/segments/audio_filename_231223150507/00-00-00.00_00-00-02.50.wav"
    },
    ...
  ]
}
```

---

## Step 2: Transcribe Segment
**Endpoint**: `/transcribe`
**Method**: `POST`
**Content-Type**: `multipart/form-data`

Transcribes a specific audio segment.

### Request Parameters
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `segment_filename` | `string` | The `filename` returned by `/segment`. |
| `file` | `File` | (Optional) Direct upload if not using pre-segmented files. |

### Response
```json
{
  "text": "Transcribed text for this segment."
}
```

---

## Integration Workflow Example (Python)

```python
import requests

# 1. Segment the long audio
files = {'file': open('long_audio.wav', 'rb')}
seg_resp = requests.post("https://0.0.0.0:8443/segment", files=files, verify=False).json()

# 2. Transcribe each segment one-by-one
full_transcript = []
for seg in seg_resp['segments']:
    data = {'segment_filename': seg['filename']}
    trans_resp = requests.post("https://0.0.0.0:8443/transcribe", data=data, verify=False).json()
    
    full_transcript.append({
        "time": f"{seg['start']}s - {seg['end']}s",
        "text": trans_resp['text']
    })
    print(f"[{seg['start']}s] {trans_resp['text']}")

# Final result is in full_transcript
```

## Large File Handling
By separating these calls, you ensure that the GPU only handles one small segment at a time, preventing CUDA OOM errors.
