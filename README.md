# GLM-ASR Nano: Professional Speech-to-Text Demo

A high-performance, robust Speech-to-Text (STT) application powered by the **GLM-ASR Nano** model. This project features a modern web interface, intelligent audio segmentation, and memory-safe processing for long audio files.

![GLM-ASR UI](https://img.shields.io/badge/UI-Modern%20Glassmorphism-blue)
![Backend](https://img.shields.io/badge/Backend-FastAPI-green)
![Model](https://img.shields.io/badge/Model-GLM--ASR%20Nano-purple)

## 🚀 Key Features

- **High Accuracy Transcription**: Utilizing the `GLM-ASR-Nano-2512` model for superior speech recognition.
- **Robust Segmentation (VAD)**: Intelligent Voice Activity Detection (Silero VAD) running on CPU to split long audio into manageable chunks.
- **Sequential GPU Processing**: Optimized workflow that transcribes segments one-by-one to prevent CUDA Out-Of-Memory (OOM) errors on systems like the RTX 4060 Ti.
- **Interactive UI**:
    - **Live Recording**: Record audio directly from your browser.
    - **File Upload**: Support for various audio formats.
    - **Smart Seek**: Click any transcript segment to jump to that specific point in the merged audio player.
    - **Glassmorphism Design**: A premium, responsive dark-mode interface built with Tailwind CSS.
- **HTTPS Enabled**: Built-in support for secure communication using self-signed certificates.

## 🛠 Tech Stack

- **Backend**: Python 3.10+, FastAPI, PyTorch, Torchaudio.
- **Frontend**: HTML5, JavaScript, Tailwind CSS (CDN).
- **VAD**: Silero VAD (CPU-based).
- **ASR Model**: GLM-ASR Nano.
- **Package Manager**: [uv](https://github.com/astral-sh/uv).

## 📥 Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd GLM-ASR
   ```

2. **Setup Virtual Environment and Dependencies**:
   This project uses `uv` for fast dependency management.
   ```bash
   uv sync
   ```

3. **Generate SSL Certificates**:
   The web UI requires HTTPS for microphone access.
   ```bash
   bash generate_certs.sh
   ```

## 🏃 Running the Application

Start the server using the provided run script:
```bash
bash run.sh
```

The application will be available at: `https://localhost:8443`

> [!NOTE]
> Since the server uses self-signed certificates, you will see a security warning in your browser. Click **Advanced** and **Proceed** to continue.

## 🔌 API Documentation

For detailed API usage (Transcribe, Segment, Full-Cycle), refer to the [STT API Documentation](stt_api_docs.md).

### Primary Endpoint
- `POST /transcribe-full`: Handles the entire process from segmentation to final transcription.

## 💻 Hardware Requirements

- **GPU**: NVIDIA GeForce RTX 4060 Ti 8GB (Recommended) or higher.
- **CUDA**: 12.2+ supported.
- **RAM**: 16GB+ recommended.

## 📜 License

[Apache 2.0](LICENSE)
