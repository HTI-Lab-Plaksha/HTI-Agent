#!/usr/bin/env bash
echo "============================================================"
echo "  Starting Meera - HTI Lab 3D Conversational Avatar Server"
echo "============================================================"
echo ""
echo "Opening browser at http://localhost:5173/web/index.html ..."

if which xdg-open > /dev/null; then
    xdg-open "http://localhost:5173/web/index.html" &
elif which open > /dev/null; then
    open "http://localhost:5173/web/index.html" &
fi

echo "Running Python Avatar Web Server on port 5173..."
python3 server.py
