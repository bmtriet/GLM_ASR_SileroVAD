from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import shutil
import os
import uuid
import gc
import torch
from asr_handler import ASRHandler
from segmentation import Segmenter
from datetime import datetime
from contextlib import asynccontextmanager

# Set environment variable before any torch operations if possible
# Though torch might already be imported, it's good practice.
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# Global handler
handler = None
segmenter = None

def get_unique_folder(filename: str) -> Path:
    """Generate folder name: [audio_file_name] + yymmddhhmmss"""
    base_name = Path(filename).stem
    timestamp = datetime.now().strftime("%y%m%d%H%M%S")
    folder_name = f"{base_name}_{timestamp}"
    full_path = Path("temp_uploads/segments") / folder_name
    full_path.mkdir(parents=True, exist_ok=True)
    return full_path

@asynccontextmanager
async def lifespan(app: FastAPI):
    global handler, segmenter
    # Initialize handler on startup
    # Assuming the checkpoint is in current directory or a known path. 
    # For now, we use current directory as default.
    checkpoint_dir = os.environ.get("CHECKPOINT_DIR", ".")
    try:
        handler = ASRHandler(checkpoint_dir=checkpoint_dir)
        print("ASR Handler initialized.")
        
        segmenter = Segmenter(handler)
        print("Segmenter initialized.")
    except Exception as e:
        print(f"Failed to initialize models: {e}")
        # We don't exit here to allow debugging if model missing
    yield
    # Clean up (if any)
    print("Shutting down...")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("temp_uploads", exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("templates/index.html", "r") as f:
        return f.read()

# Serve temp_uploads as static for downloading merged results
app.mount("/outputs", StaticFiles(directory="temp_uploads"), name="outputs")

@app.post("/segment")
async def segment_audio(
    file: UploadFile = File(...),
    min_speech_ms: int = Form(250),
    min_silence_ms: int = Form(700)
):
    if not segmenter:
        raise HTTPException(status_code=503, detail="Segmenter not initialized")
    
    file_id = str(uuid.uuid4())
    ext = file.filename.split(".")[-1]
    temp_path = f"temp_uploads/{file_id}.{ext}"
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 2. Generate unique folder for segments
        custom_dir = get_unique_folder(file.filename)
            
        # 3. Segment audio (VAD on CPU, physically splitting)
        segment_data = segmenter.segment_to_files(
            temp_path,
            min_speech_duration_ms=min_speech_ms,
            min_silence_duration_ms=min_silence_ms,
            custom_dir=custom_dir
        )
        
        segments = segment_data['segments']
        merged_path = segment_data['merged_path']
        
        return JSONResponse({
            "folder": str(custom_dir),
            "merged_audio_url": f"/outputs/{Path(merged_path).relative_to('temp_uploads')}" if merged_path else None,
            "segments": segments
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(None),
    segment_filename: str = Form(None),
):
    if not handler:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    temp_path = None
    try:
        if segment_filename:
            # Transcribe a pre-segmented file (check root and subfolders)
            temp_path = os.path.join("temp_uploads/segments", segment_filename)
            if not os.path.exists(temp_path):
                # Try finding it in subfolders if it's just a filename
                import glob
                matches = glob.glob(f"temp_uploads/segments/*/{segment_filename}")
                if matches:
                    temp_path = matches[0]
                else:
                    raise HTTPException(status_code=404, detail=f"Segment file {segment_filename} not found")
        elif file:
            # Transcribe a direct upload
            file_id = str(uuid.uuid4())
            ext = file.filename.split(".")[-1]
            temp_path = f"temp_uploads/{file_id}.{ext}"
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        else:
            raise HTTPException(status_code=400, detail="Either file or segment_filename must be provided")

        transcription = handler.transcribe(temp_path)
        return JSONResponse({"text": transcription})
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path and not segment_filename and os.path.exists(temp_path):
             os.remove(temp_path)

@app.post("/transcribe-full")
async def transcribe_full(
    file: UploadFile = File(...),
    min_speech_ms: int = Form(250),
    min_silence_ms: int = Form(700)
):
    if not handler or not segmenter:
        raise HTTPException(status_code=503, detail="Models not fully initialized")
    
    file_id = str(uuid.uuid4())
    ext = file.filename.split(".")[-1]
    temp_path = f"temp_uploads/{file_id}.{ext}"
    
    try:
        # 1. Save uploaded file
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 2. Generate unique folder for segments
        custom_dir = get_unique_folder(file.filename)
            
        # 3. Segment audio (VAD on CPU, physically splitting)
        segment_data = segmenter.segment_to_files(
            temp_path,
            min_speech_duration_ms=min_speech_ms,
            min_silence_duration_ms=min_silence_ms,
            custom_dir=custom_dir
        )
        
        segment_results = segment_data['segments']
        merged_path = segment_data['merged_path']
        
        # 4. Transcribe each segment one-by-one (GLM on GPU)
        final_segments = []
        full_text_parts = []
        
        print(f"Orchestrating transcription for {len(segment_results)} segments...")
        
        for seg in segment_results:
            seg_path = seg['path']
            try:
                text = handler.transcribe(seg_path)
                
                segment_data = {
                    "id": seg['id'],
                    "start": seg['start'],
                    "end": seg['end'],
                    "merged_start": seg.get('merged_start'),
                    "merged_end": seg.get('merged_end'),
                    "text": text
                }
                final_segments.append(segment_data)
                full_text_parts.append(text)
                
                print(f"Finished segment {seg['id']}: {seg['start']}s - {seg['end']}s")
            finally:
                # Cleanup segment file immediately
                if os.path.exists(seg_path):
                    os.remove(seg_path)
            
            # Explicit memory cleanup between segments
            torch.cuda.empty_cache()
            gc.collect()

        return JSONResponse({
            "folder": str(custom_dir),
            "merged_audio_url": f"/outputs/{Path(merged_path).relative_to('temp_uploads')}" if merged_path else None,
            "segments": final_segments,
            "full_text": " ".join(full_text_parts)
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Final cleanup of the original uploaded file
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8443, ssl_keyfile="key.pem", ssl_certfile="cert.pem")
