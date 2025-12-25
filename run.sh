#!/bin/bash

# Generate SSL certificates if they don't exist
bash generate_certs.sh

# Start the FastAPI application with HTTPS
echo "Starting GLM-ASR Demo Server on https://localhost:8443"
echo "Note: You may see a security warning due to self-signed certificate."
echo "This is normal - click 'Advanced' and 'Proceed to localhost' to continue."
echo ""
export CHECKPOINT_DIR="zai-org/GLM-ASR-Nano-2512"

source .venv/bin/activate

uvicorn app:app --reload --host 0.0.0.0 --port 8443 \
    --ssl-keyfile=key.pem \
    --ssl-certfile=cert.pem
