#!/bin/bash

# Generate self-signed SSL certificates for HTTPS
if [ ! -f "cert.pem" ] || [ ! -f "key.pem" ]; then
    echo "Generating self-signed SSL certificates..."
    openssl req -x509 -newkey rsa:4096 -nodes \
        -out cert.pem \
        -keyout key.pem \
        -days 365 \
        -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"
    echo "SSL certificates generated: cert.pem and key.pem"
else
    echo "SSL certificates already exist."
fi
