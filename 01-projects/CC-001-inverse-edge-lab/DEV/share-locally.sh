#!/bin/bash
# Quick sharing script using ngrok or localtunnel

echo "Starting Inverse Edge Lab locally..."
docker-compose up -d

echo "Waiting for app to start..."
sleep 10

echo "Choose sharing method:"
echo "1) ngrok (requires account)"
echo "2) localtunnel (no account needed)"
read -p "Enter choice (1 or 2): " choice

if [ "$choice" = "1" ]; then
    echo "Starting ngrok..."
    ngrok http 8501
else
    echo "Installing localtunnel..."
    npm install -g localtunnel
    echo "Starting localtunnel..."
    lt --port 8501 --subdomain inverse-edge-lab
fi